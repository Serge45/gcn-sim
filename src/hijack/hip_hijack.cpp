#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <elf.h>
#include "gcnsim/hip_hijack.h"
#include "gcnsim/gcnsim.h"
#include "gcnsim/debugger_core.h"


// Helper to resolve the original symbol from the next library in the link order (or via dlopen fallback)
template<typename FuncPtr>
static FuncPtr get_real_symbol(const char* name) {
    void* symbol = dlsym(RTLD_NEXT, name);
    if (!symbol) {
        static void* handle = dlopen("libamdhip64.so", RTLD_LAZY);
        if (handle) {
            symbol = dlsym(handle, name);
        }
    }
    return reinterpret_cast<FuncPtr>(symbol);
}

// ---------------------------------------------------------------------------
// Internal Debugger Integration
// ---------------------------------------------------------------------------

static bool attach_debugger_if_needed(const gcnsim::KernelInfo* ki, const gcnsim::LaunchConfig& cfg) {
    if (std::getenv("GCNSIM_DEBUG")) {
        gcnsim::Debugger db;
        db.kernel = *ki;
        db.cfg = cfg;
        db.mi_mode = (std::getenv("GCNSIM_MI_MODE") != nullptr);
        db.dwarf.load_from_elf(ki->hsaco_data, ki->hsaco_size);
        
        // Try to find a path for symbols
        if (ki->hsaco_data) {
            // Find which module this belongs to and if it has a file path
            // For now, use a simple heuristic or the host exe
            char buf[1024];
            ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf)-1);
            if (len != -1) {
                buf[len] = '\0';
                db.hsaco_path = buf;
            }
        }

        const char* state = std::getenv("GCNSIM_DB_STATE");
        if (state) db.deserialize_state(state);

        if (db.mi_mode) {
            const char* mi_fd_env = std::getenv("GCNSIM_MI_FD");
            if (mi_fd_env) {
                int mi_fd = std::stoi(mi_fd_env);
                db.set_output_fd(mi_fd);
            } else {
                // Isolate MI stream: dup stdout and redirect host stdout to stderr
                int mi_fd = dup(STDOUT_FILENO);
                dup2(STDERR_FILENO, STDOUT_FILENO);
                db.set_output_fd(mi_fd);
            }
            // In wrapper mode, the wrapper already sent the added group record.
            // But if we are running standalone, we need it.
            // frontend usually handles duplicates gracefully.
            db.mi_printf("=thread-group-added,id=\"i1\"\n");
            db.mi_printf("=thread-group-started,id=\"i1\",pid=\"%d\"\n", (int)getpid());
            db.mi_printf("=thread-created,id=\"1\",group-id=\"i1\"\n");
            db.mi_printf("*thread-selected,id=\"1\"\n");
        } else {
            std::cerr << "[gcnsim_hip] Debugger attached to kernel: " << ki->name << std::endl;
        }

        gcnsim::set_global_debugger(&db);
        db.cfg.debug_hook = gcnsim::global_debug_hook;
        db.single_stepping = true; // start paused
        
        std::string err;
        gcnsim::launch_kernel(*ki, db.cfg, err);
        return true;
    }
    return false;
}

// --- Dynamic Hijack Interception Symbols ---

extern "C" {

struct CallConfig {
    gcnsim::dim3 gridDim;
    gcnsim::dim3 blockDim;
    size_t sharedMem;
    gcnsim::hipStream_t stream;
};

static thread_local std::vector<CallConfig> g_config_stack;

void __hipPushCallConfiguration(gcnsim::dim3 gridDim, gcnsim::dim3 blockDim, size_t sharedMem, gcnsim::hipStream_t stream) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<void(*)(gcnsim::dim3, gcnsim::dim3, size_t, gcnsim::hipStream_t)>("__hipPushCallConfiguration");
        if (real_func) real_func(gridDim, blockDim, sharedMem, stream);
        return;
    }
    g_config_stack.push_back({gridDim, blockDim, sharedMem, stream});
}

void __hipPopCallConfiguration(gcnsim::dim3* gridDim, gcnsim::dim3* blockDim, size_t* sharedMem, gcnsim::hipStream_t* stream) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<void(*)(gcnsim::dim3*, gcnsim::dim3*, size_t*, gcnsim::hipStream_t*)>("__hipPopCallConfiguration");
        if (real_func) real_func(gridDim, blockDim, sharedMem, stream);
        return;
    }
    if (!g_config_stack.empty()) {
        auto cfg = g_config_stack.back();
        g_config_stack.pop_back();
        if (gridDim) *gridDim = cfg.gridDim;
        if (blockDim) *blockDim = cfg.blockDim;
        if (sharedMem) *sharedMem = cfg.sharedMem;
        if (stream) *stream = cfg.stream;
    }
}

// Fat Binary Registration Interceptors
void** __hipRegisterFatBinary(const void* data) {
    if (gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static uint64_t dummy_handle = 0xDEADC0DE;
        void** handle = reinterpret_cast<void**>(&dummy_handle);
        gcnsim::HijackManager::Get().RegisterFatBinary(handle, data);
        return handle;
    }

    static auto real_func = get_real_symbol<void**(*)(const void*)>("__hipRegisterFatBinary");
    void** handle = nullptr;
    if (real_func) {
        handle = real_func(data);
    }
    return handle;
}

void __hipRegisterFunction(
    void** fatBinHandle,
    const void* hostFun,
    char* deviceFun,
    const char* deviceName,
    int thread_limit,
    void* tid, void* bid, void* gdim, void* bdim,
    int* wsize
) {
    if (gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        gcnsim::HijackManager::Get().RegisterFunction(fatBinHandle, hostFun, deviceName);
        return;
    }

    static auto real_func = get_real_symbol<void(*)(void**, const void*, char*, const char*, int, void*, void*, void*, void*, int*)>("__hipRegisterFunction");
    if (real_func) {
        real_func(fatBinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, gdim, bdim, wsize);
    }
}

void __hipUnregisterFatBinary(void** doublePtr) {
    if (gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        return;
    }
    static auto real_func = get_real_symbol<void(*)(void**)>("__hipUnregisterFatBinary");
    if (real_func) {
        real_func(doublePtr);
    }
}

void __hipRegisterVar(
    void** fatBinHandle,
    char* hostVar,
    char* deviceVar,
    const char* deviceName,
    int ext,
    size_t size,
    int constant,
    int global
) {
    if (gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        gcnsim::HijackManager::Get().RegisterVar(fatBinHandle, hostVar, deviceVar, deviceName, ext, size, constant, global);
        return;
    }
    static auto real_func = get_real_symbol<void(*)(void**, char*, char*, const char*, int, size_t, int, int)>("__hipRegisterVar");
    if (real_func) {
        real_func(fatBinHandle, hostVar, deviceVar, deviceName, ext, size, constant, global);
    }
}

void __hipRegisterTexture(
    void** fatBinHandle,
    const void* hostVar,
    const void* deviceVar,
    const char* deviceName,
    int type,
    int norm,
    int ext
) {
    (void)fatBinHandle; (void)hostVar; (void)deviceVar; (void)deviceName; (void)type; (void)norm; (void)ext;
}

void __hipRegisterSurface(
    void** fatBinHandle,
    const void* hostVar,
    const void* deviceVar,
    const char* deviceName,
    int type,
    int ext
) {
    (void)fatBinHandle; (void)hostVar; (void)deviceVar; (void)deviceName; (void)type; (void)ext;
}

// Memory Allocation Interceptors
gcnsim::hipError_t hipMalloc(void** devPtr, size_t size) {
    std::cout << "[gcnsim_hip] hipMalloc called!" << std::endl;
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(void**, size_t)>("hipMalloc");
        if (real_func) return real_func(devPtr, size);
        return gcnsim::hipErrorLaunchFailure;
    }
    if (!devPtr) return gcnsim::hipErrorInvalidValue;
    
    *devPtr = std::malloc(size);
    return *devPtr ? gcnsim::hipSuccess : gcnsim::hipErrorMemoryAllocation;
}

gcnsim::hipError_t hipFree(void* devPtr) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(void*)>("hipFree");
        if (real_func) return real_func(devPtr);
        return gcnsim::hipErrorLaunchFailure;
    }
    if (devPtr) {
        std::free(devPtr);
    }
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipMemcpy(void* dst, const void* src, size_t size, gcnsim::hipMemcpyKind kind) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(void*, const void*, size_t, gcnsim::hipMemcpyKind)>("hipMemcpy");
        if (real_func) return real_func(dst, src, size, kind);
        return gcnsim::hipErrorLaunchFailure;
    }
    if (size > 0) {
        if (!dst || !src) return gcnsim::hipErrorInvalidValue;
        std::memcpy(dst, src, size);
    }
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipMemcpyAsync(void* dst, const void* src, size_t size, gcnsim::hipMemcpyKind kind, gcnsim::hipStream_t stream) {
    (void)stream;
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(void*, const void*, size_t, gcnsim::hipMemcpyKind, gcnsim::hipStream_t)>("hipMemcpyAsync");
        if (real_func) return real_func(dst, src, size, kind, stream);
        return gcnsim::hipErrorLaunchFailure;
    }
    return hipMemcpy(dst, src, size, kind);
}

gcnsim::hipError_t hipMemcpyHtoD(void* dst, const void* src, size_t sizeBytes) {
    return hipMemcpy(dst, src, sizeBytes, gcnsim::hipMemcpyHostToDevice);
}

gcnsim::hipError_t hipMemcpyDtoH(void* dst, const void* src, size_t sizeBytes) {
    return hipMemcpy(dst, src, sizeBytes, gcnsim::hipMemcpyDeviceToHost);
}

gcnsim::hipError_t hipMemcpyDtoD(void* dst, const void* src, size_t sizeBytes) {
    return hipMemcpy(dst, src, sizeBytes, gcnsim::hipMemcpyDeviceToDevice);
}

gcnsim::hipError_t hipMemcpyHtoDAsync(void* dst, const void* src, size_t sizeBytes, gcnsim::hipStream_t stream) {
    return hipMemcpyAsync(dst, src, sizeBytes, gcnsim::hipMemcpyHostToDevice, stream);
}

gcnsim::hipError_t hipMemcpyDtoHAsync(void* dst, const void* src, size_t sizeBytes, gcnsim::hipStream_t stream) {
    return hipMemcpyAsync(dst, src, sizeBytes, gcnsim::hipMemcpyDeviceToHost, stream);
}

gcnsim::hipError_t hipMemcpyDtoDAsync(void* dst, const void* src, size_t sizeBytes, gcnsim::hipStream_t stream) {
    return hipMemcpyAsync(dst, src, sizeBytes, gcnsim::hipMemcpyDeviceToDevice, stream);
}

static thread_local int g_current_device = 0;

struct EventInternal {
    bool recorded = false;
    uint32_t flags = 0;
    std::chrono::steady_clock::time_point time_point;
};

// Stream Interceptors
gcnsim::hipError_t hipStreamCreate(gcnsim::hipStream_t* stream) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(gcnsim::hipStream_t*)>("hipStreamCreate");
        if (real_func) return real_func(stream);
        return gcnsim::hipErrorLaunchFailure;
    }
    if (!stream) return gcnsim::hipErrorInvalidValue;
    *stream = reinterpret_cast<gcnsim::hipStream_t>(1);
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipStreamCreateWithFlags(gcnsim::hipStream_t* stream, unsigned int flags) {
    (void)flags;
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(gcnsim::hipStream_t*, unsigned int)>("hipStreamCreateWithFlags");
        if (real_func) return real_func(stream, flags);
        return gcnsim::hipErrorLaunchFailure;
    }
    return hipStreamCreate(stream);
}

gcnsim::hipError_t hipStreamDestroy(gcnsim::hipStream_t stream) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(gcnsim::hipStream_t)>("hipStreamDestroy");
        if (real_func) return real_func(stream);
        return gcnsim::hipErrorLaunchFailure;
    }
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipStreamSynchronize(gcnsim::hipStream_t stream) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(gcnsim::hipStream_t)>("hipStreamSynchronize");
        if (real_func) return real_func(stream);
        return gcnsim::hipErrorLaunchFailure;
    }
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipStreamQuery(gcnsim::hipStream_t stream) {
    (void)stream;
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipStreamWaitEvent(gcnsim::hipStream_t stream, gcnsim::hipEvent_t event, unsigned int flags) {
    (void)stream; (void)event; (void)flags;
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipDeviceSynchronize() {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)()>("hipDeviceSynchronize");
        if (real_func) return real_func();
        return gcnsim::hipErrorLaunchFailure;
    }
    return gcnsim::hipSuccess;
}

// Event Interceptors
gcnsim::hipError_t hipEventCreate(gcnsim::hipEvent_t* event) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(gcnsim::hipEvent_t*)>("hipEventCreate");
        if (real_func) return real_func(event);
        return gcnsim::hipErrorLaunchFailure;
    }
    if (!event) return gcnsim::hipErrorInvalidValue;
    auto* ev = new EventInternal();
    *event = reinterpret_cast<gcnsim::hipEvent_t>(ev);
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipEventCreateWithFlags(gcnsim::hipEvent_t* event, unsigned int flags) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(gcnsim::hipEvent_t*, unsigned int)>("hipEventCreateWithFlags");
        if (real_func) return real_func(event, flags);
        return gcnsim::hipErrorLaunchFailure;
    }
    if (!event) return gcnsim::hipErrorInvalidValue;
    auto* ev = new EventInternal();
    ev->flags = flags;
    *event = reinterpret_cast<gcnsim::hipEvent_t>(ev);
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipEventRecord(gcnsim::hipEvent_t event, gcnsim::hipStream_t stream) {
    (void)stream;
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(gcnsim::hipEvent_t, gcnsim::hipStream_t)>("hipEventRecord");
        if (real_func) return real_func(event, stream);
        return gcnsim::hipErrorLaunchFailure;
    }
    if (!event) return gcnsim::hipErrorInvalidValue;
    auto* ev = reinterpret_cast<EventInternal*>(event);
    ev->recorded = true;
    ev->time_point = std::chrono::steady_clock::now();
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipEventSynchronize(gcnsim::hipEvent_t event) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(gcnsim::hipEvent_t)>("hipEventSynchronize");
        if (real_func) return real_func(event);
        return gcnsim::hipErrorLaunchFailure;
    }
    if (!event) return gcnsim::hipErrorInvalidValue;
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipEventElapsedTime(float* ms, gcnsim::hipEvent_t start, gcnsim::hipEvent_t stop) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(float*, gcnsim::hipEvent_t, gcnsim::hipEvent_t)>("hipEventElapsedTime");
        if (real_func) return real_func(ms, start, stop);
        return gcnsim::hipErrorLaunchFailure;
    }
    if (!ms || !start || !stop) return gcnsim::hipErrorInvalidValue;
    auto* ev_start = reinterpret_cast<EventInternal*>(start);
    auto* ev_stop = reinterpret_cast<EventInternal*>(stop);
    if (!ev_start->recorded || !ev_stop->recorded) return gcnsim::hipErrorInvalidValue;
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(ev_stop->time_point - ev_start->time_point);
    *ms = static_cast<float>(dur.count()) / 1000.0f;
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipEventDestroy(gcnsim::hipEvent_t event) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(gcnsim::hipEvent_t)>("hipEventDestroy");
        if (real_func) return real_func(event);
        return gcnsim::hipErrorLaunchFailure;
    }
    if (event) {
        delete reinterpret_cast<EventInternal*>(event);
    }
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipEventQuery(gcnsim::hipEvent_t event) {
    (void)event;
    return gcnsim::hipSuccess;
}

// Device Properties & Multi-GPU Interceptors
gcnsim::hipError_t hipGetDeviceCount(int* count) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(int*)>("hipGetDeviceCount");
        if (real_func) return real_func(count);
        return gcnsim::hipErrorLaunchFailure;
    }
    if (!count) return gcnsim::hipErrorInvalidValue;
    const char* env_count = std::getenv("GCNSIM_DEVICE_COUNT");
    *count = env_count ? std::max(1, std::atoi(env_count)) : 2;
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipSetDevice(int device) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(int)>("hipSetDevice");
        if (real_func) return real_func(device);
        return gcnsim::hipErrorLaunchFailure;
    }
    if (device < 0) return gcnsim::hipErrorInvalidValue;
    int max_dev = 2;
    const char* env_count = std::getenv("GCNSIM_DEVICE_COUNT");
    if (env_count) max_dev = std::max(1, std::atoi(env_count));
    if (device >= max_dev) return gcnsim::hipErrorInvalidValue;
    g_current_device = device;
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipGetDevice(int* device) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(int*)>("hipGetDevice");
        if (real_func) return real_func(device);
        return gcnsim::hipErrorLaunchFailure;
    }
    if (!device) return gcnsim::hipErrorInvalidValue;
    *device = g_current_device;
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipGetDeviceProperties(gcnsim::hipDeviceProp_t* prop, int deviceId) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(gcnsim::hipDeviceProp_t*, int)>("hipGetDeviceProperties");
        if (real_func) return real_func(prop, deviceId);
        return gcnsim::hipErrorLaunchFailure;
    }
    if (!prop) return gcnsim::hipErrorInvalidValue;
    int max_dev = 2;
    const char* env_count = std::getenv("GCNSIM_DEVICE_COUNT");
    if (env_count) max_dev = std::max(1, std::atoi(env_count));
    if (deviceId < 0 || deviceId >= max_dev) return gcnsim::hipErrorInvalidValue;

    std::memset(prop, 0, sizeof(*prop));
    const char* env_arch = std::getenv("GCN_SIM_ARCH");
    bool is_gfx950 = (env_arch && (std::string(env_arch) == "gfx950" || std::string(env_arch) == "mi350" || std::string(env_arch) == "mi350x" || std::string(env_arch) == "mi355" || std::string(env_arch) == "mi355x"));
    bool is_gfx942 = (env_arch && (std::string(env_arch) == "gfx942" || std::string(env_arch) == "mi300" || std::string(env_arch) == "mi300x"));
    bool is_gfx908 = (env_arch && (std::string(env_arch) == "gfx908" || std::string(env_arch) == "mi100"));

    if (is_gfx950) {
        std::snprintf(prop->name, sizeof(prop->name), "AMD Instinct MI350X");
        std::strncpy(prop->gcnArchName, "gfx950:sramecc+:xnack-", sizeof(prop->gcnArchName) - 1);
        prop->totalGlobalMem = 288ULL * 1024 * 1024 * 1024; // 288 GB HBM3e
        prop->clockRate = 2100000;       // 2.1 GHz Boost in kHz
        prop->memoryClockRate = 3200000; // 3.2 GHz in kHz
        prop->multiProcessorCount = 304; // 304 CUs
        prop->l2CacheSize = 32 * 1024 * 1024; // 32 MB L2
        prop->memoryBusWidth = 8192;
        prop->major = 9;
        prop->minor = 5;
        prop->sharedMemPerBlock = 163840; // 160 KB LDS
        prop->maxSharedMemoryPerMultiProcessor = 163840;
    } else if (is_gfx942) {
        std::snprintf(prop->name, sizeof(prop->name), "AMD Instinct MI300X");
        std::strncpy(prop->gcnArchName, "gfx942:sramecc+:xnack-", sizeof(prop->gcnArchName) - 1);
        prop->totalGlobalMem = 192ULL * 1024 * 1024 * 1024; // 192 GB HBM3
        prop->clockRate = 2100000;       // 2.1 GHz Boost
        prop->memoryClockRate = 2600000; // 2.6 GHz in kHz
        prop->multiProcessorCount = 304; // 304 CUs
        prop->l2CacheSize = 32 * 1024 * 1024; // 32 MB L2
        prop->memoryBusWidth = 8192;
        prop->major = 9;
        prop->minor = 4;
        prop->sharedMemPerBlock = 65536; // 64 KB
        prop->maxSharedMemoryPerMultiProcessor = 65536;
    } else if (is_gfx908) {
        std::snprintf(prop->name, sizeof(prop->name), "AMD Instinct MI100");
        std::strncpy(prop->gcnArchName, "gfx908:sramecc+:xnack-", sizeof(prop->gcnArchName) - 1);
        prop->totalGlobalMem = 32ULL * 1024 * 1024 * 1024; // 32 GB HBM2
        prop->clockRate = 1502000;       // 1.502 GHz Boost in kHz
        prop->memoryClockRate = 1200000; // 1.2 GHz in kHz
        prop->multiProcessorCount = 120; // 120 CUs
        prop->l2CacheSize = 8 * 1024 * 1024; // 8 MB L2
        prop->memoryBusWidth = 4096;     // 4096-bit HBM2
        prop->major = 9;
        prop->minor = 0;
        prop->sharedMemPerBlock = 65536; // 64 KB
        prop->maxSharedMemoryPerMultiProcessor = 65536;
    } else {
        std::snprintf(prop->name, sizeof(prop->name), "AMD Instinct MI250X (GCD %d)", deviceId);
        std::strncpy(prop->gcnArchName, "gfx90a:sramecc+:xnack-", sizeof(prop->gcnArchName) - 1);
        prop->totalGlobalMem = 64ULL * 1024 * 1024 * 1024; // 64 GB HBM2e per GCD
        prop->clockRate = 1700000;       // 1.7 GHz in kHz
        prop->memoryClockRate = 1600000; // 1.6 GHz in kHz
        prop->multiProcessorCount = 110; // 110 CUs per GCD
        prop->l2CacheSize = 8 * 1024 * 1024; // 8 MB L2 per GCD
        prop->memoryBusWidth = 8192;
        prop->major = 9;
        prop->minor = 0;
        prop->sharedMemPerBlock = 65536; // 64 KB
        prop->maxSharedMemoryPerMultiProcessor = 65536;
    }
    prop->regsPerBlock = 65536;
    prop->warpSize = 64;
    prop->maxThreadsPerBlock = 1024;
    prop->maxThreadsDim[0] = 1024;
    prop->maxThreadsDim[1] = 1024;
    prop->maxThreadsDim[2] = 1024;
    prop->maxGridSize[0] = 2147483647;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;
    prop->totalConstMem = 2147483647;
    prop->maxThreadsPerMultiProcessor = 2048;
    prop->computeMode = 0;
    prop->concurrentKernels = 1;
    prop->pciDomainID = 0;
    prop->pciBusID = 3;
    prop->pciDeviceID = deviceId;
    prop->isMultiGpuBoard = 1;
    prop->canMapHostMemory = 1;
    prop->integrated = 0;
    prop->cooperativeLaunch = 1;
    prop->cooperativeMultiDeviceLaunch = 1;
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipGetDevicePropertiesR0600(gcnsim::hipDeviceProp_t* prop, int deviceId) {
    return hipGetDeviceProperties(prop, deviceId);
}

gcnsim::hipError_t hipGetDevicePropertiesR0000(gcnsim::hipDeviceProp_t* prop, int deviceId) {
    return hipGetDeviceProperties(prop, deviceId);
}

const char* hipGetErrorString(gcnsim::hipError_t hip_error) {
    switch (hip_error) {
        case gcnsim::hipSuccess: return "hipSuccess";
        case gcnsim::hipErrorInvalidValue: return "hipErrorInvalidValue";
        case gcnsim::hipErrorOutOfMemory: return "hipErrorOutOfMemory";
        case gcnsim::hipErrorInvalidDeviceFunction: return "hipErrorInvalidDeviceFunction";
        case gcnsim::hipErrorLaunchFailure: return "hipErrorLaunchFailure";
        case gcnsim::hipErrorNotFound: return "hipErrorNotFound";
        case gcnsim::hipErrorInvalidImage: return "hipErrorInvalidImage";
        case gcnsim::hipErrorInvalidSymbol: return "hipErrorInvalidSymbol";
        default: return "hipErrorUnknown";
    }
}

gcnsim::hipError_t hipDeviceGetAttribute(int* pi, gcnsim::hipDeviceAttribute_t attr, int deviceId) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(int*, gcnsim::hipDeviceAttribute_t, int)>("hipDeviceGetAttribute");
        if (real_func) return real_func(pi, attr, deviceId);
        return gcnsim::hipErrorLaunchFailure;
    }
    if (!pi) return gcnsim::hipErrorInvalidValue;
    int max_dev = 2;
    const char* env_count = std::getenv("GCNSIM_DEVICE_COUNT");
    if (env_count) max_dev = std::max(1, std::atoi(env_count));
    if (deviceId < 0 || deviceId >= max_dev) return gcnsim::hipErrorInvalidValue;

    switch (attr) {
        case gcnsim::hipDeviceAttributeMaxThreadsPerBlock: *pi = 1024; break;
        case gcnsim::hipDeviceAttributeMaxBlockDimX: *pi = 1024; break;
        case gcnsim::hipDeviceAttributeMaxBlockDimY: *pi = 1024; break;
        case gcnsim::hipDeviceAttributeMaxBlockDimZ: *pi = 1024; break;
        case gcnsim::hipDeviceAttributeMaxGridDimX: *pi = 2147483647; break;
        case gcnsim::hipDeviceAttributeMaxGridDimY: *pi = 65535; break;
        case gcnsim::hipDeviceAttributeMaxGridDimZ: *pi = 65535; break;
        case gcnsim::hipDeviceAttributeMaxSharedMemoryPerBlock: *pi = 65536; break;
        case gcnsim::hipDeviceAttributeTotalConstantMemory: *pi = 2147483647; break;
        case gcnsim::hipDeviceAttributeWarpSize: *pi = 64; break;
        case gcnsim::hipDeviceAttributeMaxRegistersPerBlock: *pi = 65536; break;
        case gcnsim::hipDeviceAttributeClockRate: *pi = 1700000; break;
        case gcnsim::hipDeviceAttributeMemoryClockRate: *pi = 1600000; break;
        case gcnsim::hipDeviceAttributeMemoryBusWidth: *pi = 8192; break;
        case gcnsim::hipDeviceAttributeMultiprocessorCount: *pi = 110; break;
        case gcnsim::hipDeviceAttributeComputeMode: *pi = 0; break;
        case gcnsim::hipDeviceAttributeL2CacheSize: *pi = 8 * 1024 * 1024; break;
        case gcnsim::hipDeviceAttributeMaxThreadsPerMultiProcessor: *pi = 2048; break;
        case gcnsim::hipDeviceAttributeComputeCapabilityMajor: *pi = 9; break;
        case gcnsim::hipDeviceAttributeComputeCapabilityMinor: *pi = 0; break;
        case gcnsim::hipDeviceAttributeConcurrentKernels: *pi = 1; break;
        case gcnsim::hipDeviceAttributePciBusId: *pi = 3; break;
        case gcnsim::hipDeviceAttributePciDeviceId: *pi = deviceId; break;
        case gcnsim::hipDeviceAttributeMaxSharedMemoryPerMultiprocessor: *pi = 65536; break;
        case gcnsim::hipDeviceAttributeIsMultiGpuBoard: *pi = 1; break;
        case gcnsim::hipDeviceAttributeIntegrated: *pi = 0; break;
        case gcnsim::hipDeviceAttributeCooperativeLaunch: *pi = 1; break;
        case gcnsim::hipDeviceAttributeCooperativeMultiDeviceLaunch: *pi = 1; break;
        default: *pi = 0; break;
    }
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(int*, int, int)>("hipDeviceCanAccessPeer");
        if (real_func) return real_func(canAccessPeer, deviceId, peerDeviceId);
        return gcnsim::hipErrorLaunchFailure;
    }
    if (!canAccessPeer) return gcnsim::hipErrorInvalidValue;
    *canAccessPeer = (deviceId != peerDeviceId) ? 1 : 0;
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags) {
    (void)peerDeviceId; (void)flags;
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipDeviceDisablePeerAccess(int peerDeviceId) {
    (void)peerDeviceId;
    return gcnsim::hipSuccess;
}

// Memory Operations (Memset, HostAlloc, PeerMemcpy, 2D)
gcnsim::hipError_t hipMemset(void* devPtr, int value, size_t count) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(void*, int, size_t)>("hipMemset");
        if (real_func) return real_func(devPtr, value, count);
        return gcnsim::hipErrorLaunchFailure;
    }
    if (count > 0) {
        if (!devPtr) return gcnsim::hipErrorInvalidValue;
        std::memset(devPtr, value, count);
    }
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipMemsetAsync(void* devPtr, int value, size_t count, gcnsim::hipStream_t stream) {
    (void)stream;
    return hipMemset(devPtr, value, count);
}

gcnsim::hipError_t hipMemsetD8(void* devPtr, unsigned char value, size_t count) {
    return hipMemset(devPtr, value, count);
}

gcnsim::hipError_t hipMemsetD16(void* devPtr, unsigned short value, size_t count) {
    if (!devPtr) return gcnsim::hipErrorInvalidValue;
    unsigned short* p = reinterpret_cast<unsigned short*>(devPtr);
    for (size_t i = 0; i < count; i++) p[i] = value;
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipMemsetD32(void* devPtr, int value, size_t count) {
    if (!devPtr) return gcnsim::hipErrorInvalidValue;
    int* p = reinterpret_cast<int*>(devPtr);
    for (size_t i = 0; i < count; i++) p[i] = value;
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipMemsetD8Async(void* devPtr, unsigned char value, size_t count, gcnsim::hipStream_t stream) {
    (void)stream;
    return hipMemsetD8(devPtr, value, count);
}

gcnsim::hipError_t hipMemsetD16Async(void* devPtr, unsigned short value, size_t count, gcnsim::hipStream_t stream) {
    (void)stream;
    return hipMemsetD16(devPtr, value, count);
}

gcnsim::hipError_t hipMemsetD32Async(void* devPtr, int value, size_t count, gcnsim::hipStream_t stream) {
    (void)stream;
    return hipMemsetD32(devPtr, value, count);
}

gcnsim::hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags) {
    (void)flags;
    return hipMalloc(ptr, size);
}

gcnsim::hipError_t hipHostFree(void* ptr) {
    return hipFree(ptr);
}

gcnsim::hipError_t hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags) {
    (void)flags;
    if (!devPtr || !hstPtr) return gcnsim::hipErrorInvalidValue;
    *devPtr = hstPtr;
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipPointerGetAttributes(gcnsim::hipPointerAttribute_t* attributes, const void* ptr) {
    if (!attributes || !ptr) return gcnsim::hipErrorInvalidValue;
    attributes->memoryType = gcnsim::hipMemoryTypeDevice;
    attributes->device = g_current_device;
    attributes->devicePointer = const_cast<void*>(ptr);
    attributes->hostPointer = const_cast<void*>(ptr);
    attributes->isManaged = 0;
    attributes->allocationID = 1;
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipMemcpyPeer(void* dst, int dstDeviceId, const void* src, int srcDeviceId, size_t size) {
    (void)dstDeviceId; (void)srcDeviceId;
    return hipMemcpy(dst, src, size, gcnsim::hipMemcpyDeviceToDevice);
}

gcnsim::hipError_t hipMemcpyPeerAsync(void* dst, int dstDeviceId, const void* src, int srcDeviceId, size_t size, gcnsim::hipStream_t stream) {
    (void)dstDeviceId; (void)srcDeviceId; (void)stream;
    return hipMemcpy(dst, src, size, gcnsim::hipMemcpyDeviceToDevice);
}

gcnsim::hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, gcnsim::hipMemcpyKind kind) {
    (void)kind;
    if (!dst || !src) return gcnsim::hipErrorInvalidValue;
    for (size_t y = 0; y < height; y++) {
        std::memcpy(static_cast<char*>(dst) + y * dpitch, static_cast<const char*>(src) + y * spitch, width);
    }
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, gcnsim::hipMemcpyKind kind, gcnsim::hipStream_t stream) {
    (void)stream;
    return hipMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
}

gcnsim::hipError_t hipGetSymbolAddress(void** devPtr, const void* symbol) {
    if (!devPtr || !symbol) return gcnsim::hipErrorInvalidValue;
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(void**, const void*)>("hipGetSymbolAddress");
        if (real_func) return real_func(devPtr, symbol);
        return gcnsim::hipErrorLaunchFailure;
    }
    const auto* var = gcnsim::HijackManager::Get().FindRegisteredVar(symbol);
    if (var && var->dev_ptr) {
        *devPtr = var->dev_ptr;
    } else {
        *devPtr = const_cast<void*>(symbol);
    }
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipGetSymbolSize(size_t* size, const void* symbol) {
    if (!size || !symbol) return gcnsim::hipErrorInvalidValue;
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(size_t*, const void*)>("hipGetSymbolSize");
        if (real_func) return real_func(size, symbol);
        return gcnsim::hipErrorLaunchFailure;
    }
    const auto* var = gcnsim::HijackManager::Get().FindRegisteredVar(symbol);
    if (var) {
        *size = var->size;
        return gcnsim::hipSuccess;
    }
    return gcnsim::hipErrorInvalidSymbol;
}

gcnsim::hipError_t hipMemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset, gcnsim::hipMemcpyKind kind) {
    (void)kind;
    if (!symbol || !src) return gcnsim::hipErrorInvalidValue;
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(const void*, const void*, size_t, size_t, gcnsim::hipMemcpyKind)>("hipMemcpyToSymbol");
        if (real_func) return real_func(symbol, src, count, offset, kind);
        return gcnsim::hipErrorLaunchFailure;
    }
    const auto* var = gcnsim::HijackManager::Get().FindRegisteredVar(symbol);
    void* dst_target = nullptr;
    if (var && var->dev_ptr) {
        if (offset + count > var->size) return gcnsim::hipErrorInvalidValue;
        dst_target = static_cast<char*>(var->dev_ptr) + offset;
        std::memcpy(dst_target, src, count);
        if (var->host_var) {
            std::memcpy(static_cast<char*>(const_cast<void*>(var->host_var)) + offset, src, count);
        }
    } else {
        dst_target = static_cast<char*>(const_cast<void*>(symbol)) + offset;
        std::memcpy(dst_target, src, count);
    }
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipMemcpyToSymbolAsync(const void* symbol, const void* src, size_t count, size_t offset, gcnsim::hipMemcpyKind kind, gcnsim::hipStream_t stream) {
    (void)stream;
    return hipMemcpyToSymbol(symbol, src, count, offset, kind);
}

gcnsim::hipError_t hipMemcpyFromSymbol(void* dst, const void* symbol, size_t count, size_t offset, gcnsim::hipMemcpyKind kind) {
    (void)kind;
    if (!dst || !symbol) return gcnsim::hipErrorInvalidValue;
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(void*, const void*, size_t, size_t, gcnsim::hipMemcpyKind)>("hipMemcpyFromSymbol");
        if (real_func) return real_func(dst, symbol, count, offset, kind);
        return gcnsim::hipErrorLaunchFailure;
    }
    const auto* var = gcnsim::HijackManager::Get().FindRegisteredVar(symbol);
    const void* src_target = nullptr;
    if (var && var->dev_ptr) {
        if (offset + count > var->size) return gcnsim::hipErrorInvalidValue;
        src_target = static_cast<const char*>(var->dev_ptr) + offset;
    } else {
        src_target = static_cast<const char*>(symbol) + offset;
    }
    std::memcpy(dst, src_target, count);
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t count, size_t offset, gcnsim::hipMemcpyKind kind, gcnsim::hipStream_t stream) {
    (void)stream;
    return hipMemcpyFromSymbol(dst, symbol, count, offset, kind);
}

gcnsim::hipError_t hipGetLastError(void) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(void)>("hipGetLastError");
        if (real_func) return real_func();
        return gcnsim::hipErrorLaunchFailure;
    }
    return gcnsim::hipSuccess;
}

// Runtime API Launch Interceptor
gcnsim::hipError_t hipLaunchKernel(
    const void* function_address,
    gcnsim::dim3 numBlocks,
    gcnsim::dim3 dimBlocks,
    void** args,
    size_t sharedMemBytes,
    gcnsim::hipStream_t stream
) {
    std::cout << "[gcnsim_hip] hipLaunchKernel called!" << std::endl;
    (void)sharedMemBytes;
    (void)stream;

    auto* reg = gcnsim::HijackManager::Get().GetRegisteredKernel(function_address);
    std::string k_name = reg ? reg->name : "";

    if (!gcnsim::HijackManager::Get().ShouldSimulateKernel(k_name)) {
        static auto real_launch = get_real_symbol<gcnsim::hipError_t(*)(const void*, gcnsim::dim3, gcnsim::dim3, void**, size_t, gcnsim::hipStream_t)>("hipLaunchKernel");
        if (real_launch) {
            return real_launch(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream);
        }
        return gcnsim::hipErrorLaunchFailure;
    }

    const gcnsim::KernelInfo* ki = gcnsim::HijackManager::Get().ResolveKernelInfo(*reg);
    if (!ki) {
        std::cerr << "[gcnsim_hip] ResolveKernelInfo failed for kernel: " << k_name << std::endl;
        return gcnsim::hipErrorInvalidDeviceFunction;
    }
    std::cout << "[gcnsim_hip] ResolveKernelInfo succeeded for kernel: " << k_name << "!" << std::endl;

    // Pack arguments using metadata offsets and sizes
    uint32_t kernarg_size = ki->descriptor ? ki->descriptor->kernarg_size : 0;
    if (kernarg_size == 0) {
        for (const auto& arg : ki->args) {
            if (arg.offset + arg.size > kernarg_size) {
                kernarg_size = arg.offset + arg.size;
            }
        }
        if (kernarg_size == 0) kernarg_size = 256;
    }
    std::vector<uint8_t> kernarg_buffer(kernarg_size, 0);
    size_t param_idx = 0;
    for (const auto& arg : ki->args) {
        if (arg.value_kind.rfind("hidden_", 0) == 0) {
            if (arg.value_kind == "hidden_block_count_x") {
                uint32_t val = numBlocks.x;
                std::memcpy(kernarg_buffer.data() + arg.offset, &val, arg.size);
            } else if (arg.value_kind == "hidden_block_count_y") {
                uint32_t val = numBlocks.y;
                std::memcpy(kernarg_buffer.data() + arg.offset, &val, arg.size);
            } else if (arg.value_kind == "hidden_block_count_z") {
                uint32_t val = numBlocks.z;
                std::memcpy(kernarg_buffer.data() + arg.offset, &val, arg.size);
            } else if (arg.value_kind == "hidden_group_size_x") {
                uint16_t val = dimBlocks.x;
                std::memcpy(kernarg_buffer.data() + arg.offset, &val, arg.size);
            } else if (arg.value_kind == "hidden_group_size_y") {
                uint16_t val = dimBlocks.y;
                std::memcpy(kernarg_buffer.data() + arg.offset, &val, arg.size);
            } else if (arg.value_kind == "hidden_group_size_z") {
                uint16_t val = dimBlocks.z;
                std::memcpy(kernarg_buffer.data() + arg.offset, &val, arg.size);
            }
            continue;
        }
        if (args && args[param_idx]) {
            std::memcpy(kernarg_buffer.data() + arg.offset, args[param_idx], arg.size);
        }
        param_idx++;
    }

    // Build gcnsim launch configuration (threads count = blocks * threads_per_block)
    gcnsim::LaunchConfig cfg;
    cfg.grid_x = numBlocks.x * dimBlocks.x;
    cfg.grid_y = numBlocks.y * dimBlocks.y;
    cfg.grid_z = numBlocks.z * dimBlocks.z;
    cfg.block_x = dimBlocks.x;
    cfg.block_y = dimBlocks.y;
    cfg.block_z = dimBlocks.z;
    cfg.kernarg = kernarg_buffer.data();
    cfg.kernarg_size = kernarg_buffer.size();

    if (attach_debugger_if_needed(ki, cfg)) {
        return gcnsim::hipSuccess;
    }

    std::string err;
    bool ok = gcnsim::launch_kernel(*ki, cfg, err);
    if (!ok) {
        std::cerr << "[gcnsim_hip] launch_kernel failed: " << err << std::endl;
        return gcnsim::hipErrorLaunchFailure;
    }
    return gcnsim::hipSuccess;
}

// Driver API Launch Interceptor
gcnsim::hipError_t hipModuleLaunchKernel(
    gcnsim::hipFunction_t f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes, gcnsim::hipStream_t stream,
    void** kernelParams, void** extra
) {
    return hipExtModuleLaunchKernel(
        f,
        gridDimX * blockDimX, gridDimY * blockDimY, gridDimZ * blockDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes, stream, kernelParams, extra,
        nullptr, nullptr, 0
    );
}

// Driver Extension Launch Interceptor (the primary driver launcher)
gcnsim::hipError_t hipExtModuleLaunchKernel(
    gcnsim::hipFunction_t f,
    uint32_t globalWorkSizeX, uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
    uint32_t localWorkSizeX,  uint32_t localWorkSizeY,  uint32_t localWorkSizeZ,
    size_t sharedMemBytes, gcnsim::hipStream_t hStream,
    void** kernelParams, void** extra,
    gcnsim::hipEvent_t startEvent, gcnsim::hipEvent_t stopEvent,
    uint32_t flags
) {
    (void)sharedMemBytes;
    (void)hStream;
    (void)startEvent;
    (void)stopEvent;
    (void)flags;

    // Wait: if dynamic interception is globally disabled, call the real one!
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_launch = get_real_symbol<gcnsim::hipError_t(*)(
            gcnsim::hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
            size_t, gcnsim::hipStream_t, void**, void**, gcnsim::hipEvent_t, gcnsim::hipEvent_t, uint32_t
        )>("hipExtModuleLaunchKernel");
        if (real_launch) {
            return real_launch(f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
                               localWorkSizeX, localWorkSizeY, localWorkSizeZ,
                               sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent, flags);
        }
        return gcnsim::hipErrorLaunchFailure;
    }

    // When dynamic hijack is enabled, f is a pointer to the KernelInfo directly, 
    // or to our SimulatedFunction representation (depending on how module loading was hijacked).
    // Let's make it robust: try to cast directly to const gcnsim::KernelInfo* first.
    const gcnsim::KernelInfo* ki = reinterpret_cast<const gcnsim::KernelInfo*>(f);
    if (!ki || ki->name.empty()) {
        return gcnsim::hipErrorInvalidDeviceFunction;
    }

    // Build gcnsim launch configuration (globalWorkSize is already total threads)
    gcnsim::LaunchConfig cfg;
    cfg.grid_x  = globalWorkSizeX;
    cfg.grid_y  = globalWorkSizeY;
    cfg.grid_z  = globalWorkSizeZ;
    cfg.block_x = localWorkSizeX;
    cfg.block_y = localWorkSizeY;
    cfg.block_z = localWorkSizeZ;

    std::vector<uint8_t> unpacked_buffer;

    if (kernelParams) {
        // Mode A: unpack using metadata offsets/sizes
        uint32_t kernarg_size = ki->descriptor ? ki->descriptor->kernarg_size : 0;
        if (kernarg_size == 0) {
            for (const auto& arg : ki->args) {
                if (arg.offset + arg.size > kernarg_size) {
                    kernarg_size = arg.offset + arg.size;
                }
            }
            if (kernarg_size == 0) kernarg_size = 256;
        }
        unpacked_buffer.assign(kernarg_size, 0);
        size_t param_idx = 0;
        for (const auto& arg : ki->args) {
            if (arg.value_kind.rfind("hidden_", 0) == 0) {
                if (arg.value_kind == "hidden_block_count_x") {
                    uint32_t val = globalWorkSizeX / localWorkSizeX;
                    std::memcpy(unpacked_buffer.data() + arg.offset, &val, arg.size);
                } else if (arg.value_kind == "hidden_block_count_y") {
                    uint32_t val = globalWorkSizeY / localWorkSizeY;
                    std::memcpy(unpacked_buffer.data() + arg.offset, &val, arg.size);
                } else if (arg.value_kind == "hidden_block_count_z") {
                    uint32_t val = globalWorkSizeZ / localWorkSizeZ;
                    std::memcpy(unpacked_buffer.data() + arg.offset, &val, arg.size);
                } else if (arg.value_kind == "hidden_group_size_x") {
                    uint16_t val = localWorkSizeX;
                    std::memcpy(unpacked_buffer.data() + arg.offset, &val, arg.size);
                } else if (arg.value_kind == "hidden_group_size_y") {
                    uint16_t val = localWorkSizeY;
                    std::memcpy(unpacked_buffer.data() + arg.offset, &val, arg.size);
                } else if (arg.value_kind == "hidden_group_size_z") {
                    uint16_t val = localWorkSizeZ;
                    std::memcpy(unpacked_buffer.data() + arg.offset, &val, arg.size);
                }
                continue;
            }
            if (kernelParams[param_idx]) {
                std::memcpy(unpacked_buffer.data() + arg.offset, kernelParams[param_idx], arg.size);
            }
            param_idx++;
        }
        cfg.kernarg = unpacked_buffer.data();
        cfg.kernarg_size = unpacked_buffer.size();
    } else if (extra) {
        // Mode B: parse extra arguments list
        void* packed_buffer = nullptr;
        size_t packed_size = 0;

        for (size_t i = 0; extra[i] != nullptr && extra[i] != HIP_LAUNCH_PARAM_END; ) {
            if (extra[i] == HIP_LAUNCH_PARAM_BUFFER_POINTER) {
                packed_buffer = extra[i + 1];
                i += 2;
            } else if (extra[i] == HIP_LAUNCH_PARAM_BUFFER_SIZE) {
                uintptr_t val = reinterpret_cast<uintptr_t>(extra[i + 1]);
                if (val < 1048576) {
                    packed_size = static_cast<size_t>(val);
                } else {
                    packed_size = *reinterpret_cast<const size_t*>(extra[i + 1]);
                }
                i += 2;
            } else {
                i++;
            }
        }
        cfg.kernarg = packed_buffer;
        cfg.kernarg_size = packed_size;
    } else {
        return gcnsim::hipErrorInvalidValue;
    }

    if (attach_debugger_if_needed(ki, cfg)) {
        return gcnsim::hipSuccess;
    }

    std::string err;
    bool ok = gcnsim::launch_kernel(*ki, cfg, err);
    if (!ok) {
        std::cerr << "[gcnsim_hip] launch_kernel failed: " << err << std::endl;
        return gcnsim::hipErrorLaunchFailure;
    }

    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipModuleLoad(gcnsim::hipModule_t* module, const char* fname) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(gcnsim::hipModule_t*, const char*)>("hipModuleLoad");
        if (real_func) return real_func(module, fname);
        return gcnsim::hipErrorLaunchFailure;
    }
    if (!module || !fname) return gcnsim::hipErrorInvalidValue;

    std::string err;
    std::string path_to_load = fname;
    auto mod = gcnsim::load_hsaco(path_to_load, err);
    if (!mod && !path_to_load.empty() && path_to_load[0] != '/') {
        std::string err2;
        std::string try_path = "../" + path_to_load;
        mod = gcnsim::load_hsaco(try_path, err2);
        if (!mod) {
            try_path = "../../" + path_to_load;
            mod = gcnsim::load_hsaco(try_path, err2);
        }
        if (mod) {
            path_to_load = try_path;
        }
    }

    if (!mod) {
        std::cerr << "[gcnsim_hip] hipModuleLoad failed to load " << fname << ": " << err << std::endl;
        return gcnsim::hipErrorNotFound;
    }

    auto* heap_mod = new gcnsim::HsacoModule(std::move(*mod));
    *module = reinterpret_cast<gcnsim::hipModule_t>(heap_mod);
    return gcnsim::hipSuccess;
}

static size_t get_inmemory_elf_size(const void* image) {
    if (!image) return 0;
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(image);
    if (bytes[0] != 0x7f || bytes[1] != 'E' || bytes[2] != 'L' || bytes[3] != 'F') {
        return 0;
    }
    const Elf64_Ehdr* ehdr = reinterpret_cast<const Elf64_Ehdr*>(image);
    size_t max_size = sizeof(Elf64_Ehdr);
    if (ehdr->e_phoff > 0) {
        max_size = std::max(max_size, (size_t)(ehdr->e_phoff + ehdr->e_phentsize * ehdr->e_phnum));
    }
    if (ehdr->e_shoff > 0) {
        max_size = std::max(max_size, (size_t)(ehdr->e_shoff + ehdr->e_shentsize * ehdr->e_shnum));
        const Elf64_Shdr* shdrs = reinterpret_cast<const Elf64_Shdr*>(bytes + ehdr->e_shoff);
        for (uint16_t i = 0; i < ehdr->e_shnum; ++i) {
            if (shdrs[i].sh_type != SHT_NOBITS) {
                max_size = std::max(max_size, (size_t)(shdrs[i].sh_offset + shdrs[i].sh_size));
            }
        }
    }
    return max_size;
}

gcnsim::hipError_t hipModuleLoadData(gcnsim::hipModule_t* module, const void* image) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(gcnsim::hipModule_t*, const void*)>("hipModuleLoadData");
        if (real_func) return real_func(module, image);
        return gcnsim::hipErrorLaunchFailure;
    }
    if (!module || !image) return gcnsim::hipErrorInvalidValue;

    size_t size = get_inmemory_elf_size(image);
    if (size == 0) return gcnsim::hipErrorInvalidImage;

    std::string err;
    auto mod = gcnsim::load_hsaco_from_memory(reinterpret_cast<const uint8_t*>(image), size, err);
    if (!mod) {
        std::cerr << "[gcnsim_hip] hipModuleLoadData failed: " << err << std::endl;
        return gcnsim::hipErrorInvalidImage;
    }

    auto* heap_mod = new gcnsim::HsacoModule(std::move(*mod));
    *module = reinterpret_cast<gcnsim::hipModule_t>(heap_mod);
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipModuleLoadDataEx(
    gcnsim::hipModule_t* module,
    const void* image,
    unsigned int numOptions,
    gcnsim::hipJitOption* options,
    void** optionValues
) {
    (void)numOptions; (void)options; (void)optionValues;
    return hipModuleLoadData(module, image);
}

gcnsim::hipError_t hipModuleGetFunction(gcnsim::hipFunction_t* function, gcnsim::hipModule_t module, const char* name) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(gcnsim::hipFunction_t*, gcnsim::hipModule_t, const char*)>("hipModuleGetFunction");
        if (real_func) return real_func(function, module, name);
        return gcnsim::hipErrorLaunchFailure;
    }
    if (!function || !module || !name) return gcnsim::hipErrorInvalidValue;

    auto* heap_mod = reinterpret_cast<gcnsim::HsacoModule*>(module);
    for (const auto& k : heap_mod->kernels) {
        if (k.name == name) {
            *function = reinterpret_cast<gcnsim::hipFunction_t>(const_cast<gcnsim::KernelInfo*>(&k));
            return gcnsim::hipSuccess;
        }
    }

    std::cerr << "[gcnsim_hip] hipModuleGetFunction: kernel " << name << " not found in module" << std::endl;
    return gcnsim::hipErrorNotFound;
}

gcnsim::hipError_t hipModuleUnload(gcnsim::hipModule_t module) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(gcnsim::hipModule_t)>("hipModuleUnload");
        if (real_func) return real_func(module);
        return gcnsim::hipErrorLaunchFailure;
    }
    if (module) {
        auto* heap_mod = reinterpret_cast<gcnsim::HsacoModule*>(module);
        delete heap_mod;
    }
    return gcnsim::hipSuccess;
}

uint16_t __gnu_f2h_ieee(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000;
        uint32_t t = static_cast<uint32_t>(14 - exp);
        uint32_t round_bit = 1 << (t - 1);
        mant = (mant + round_bit) >> t;
        return static_cast<uint16_t>(sign | mant);
    } else if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00 | (mant ? 0x200 : 0));
    }
    uint32_t round_bit = 0x00001000;
    mant = (mant + round_bit - 1 + ((mant >> 13) & 1)) >> 13;
    if (mant > 0x3ff) {
        mant = 0;
        exp++;
    }
    return static_cast<uint16_t>(sign | (exp << 10) | mant);
}

float __gnu_h2f_ieee(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t val;
    if (exp == 0) {
        if (mant == 0) {
            val = sign;
        } else {
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            exp++;
            mant &= ~0x400;
            val = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        val = sign | 0x7f800000 | (mant << 13);
    } else {
        val = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &val, 4);
    return f;
}

} // extern "C"
