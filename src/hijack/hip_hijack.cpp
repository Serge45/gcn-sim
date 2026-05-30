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

// Stream and Event Mocks (Bypassed)
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

gcnsim::hipError_t hipDeviceSynchronize() {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)()>("hipDeviceSynchronize");
        if (real_func) return real_func();
        return gcnsim::hipErrorLaunchFailure;
    }
    return gcnsim::hipSuccess;
}

// Device Properties Interceptors
gcnsim::hipError_t hipGetDeviceCount(int* count) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(int*)>("hipGetDeviceCount");
        if (real_func) return real_func(count);
        return gcnsim::hipErrorLaunchFailure;
    }
    if (!count) return gcnsim::hipErrorInvalidValue;
    *count = 1;
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipSetDevice(int device) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(int)>("hipSetDevice");
        if (real_func) return real_func(device);
        return gcnsim::hipErrorLaunchFailure;
    }
    return gcnsim::hipSuccess;
}

gcnsim::hipError_t hipGetDevice(int* device) {
    if (!gcnsim::HijackManager::Get().IsGlobalEnabled()) {
        static auto real_func = get_real_symbol<gcnsim::hipError_t(*)(int*)>("hipGetDevice");
        if (real_func) return real_func(device);
        return gcnsim::hipErrorLaunchFailure;
    }
    if (!device) return gcnsim::hipErrorInvalidValue;
    *device = 0;
    return gcnsim::hipSuccess;
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

        for (size_t i = 0; extra[i] != HIP_LAUNCH_PARAM_END; ) {
            if (extra[i] == HIP_LAUNCH_PARAM_BUFFER_POINTER) {
                packed_buffer = extra[i + 1];
                i += 2;
            } else if (extra[i] == HIP_LAUNCH_PARAM_BUFFER_SIZE) {
                packed_size = *reinterpret_cast<size_t*>(extra[i + 1]);
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

} // extern "C"
