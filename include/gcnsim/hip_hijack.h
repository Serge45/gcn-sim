#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cctype>
#include "elf_loader.h"

namespace gcnsim {

// --- Mocked HIP Types (to keep this portable and independent of host ROCm installs) ---

enum hipError_t {
    hipSuccess = 0,
    hipErrorInvalidValue = 1,
    hipErrorOutOfMemory = 2,
    hipErrorMemoryAllocation = 2,
    hipErrorInvalidDeviceFunction = 8,
    hipErrorInvalidSymbol = 13,
    hipErrorLaunchFailure = 901,
    hipErrorNotFound = 300,
    hipErrorInvalidImage = 200,
};

struct dim3 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    dim3(uint32_t x = 1, uint32_t y = 1, uint32_t z = 1) : x(x), y(y), z(z) {}
};

typedef void* hipStream_t;
typedef void* hipEvent_t;
typedef void* hipFunction_t;
typedef void* hipModule_t;

enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4
};

enum hipMemoryType {
    hipMemoryTypeHost = 1,
    hipMemoryTypeDevice = 2,
    hipMemoryTypeArray = 3,
    hipMemoryTypeUnified = 4
};

struct hipPointerAttribute_t {
    hipMemoryType memoryType;
    int device;
    void* devicePointer;
    void* hostPointer;
    int isManaged;
    unsigned allocationID;
};

struct hipDeviceProp_t {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    int memoryClockRate;
    int memoryBusWidth;
    size_t totalConstMem;
    int major;
    int minor;
    int multiProcessorCount;
    int l2CacheSize;
    int maxThreadsPerMultiProcessor;
    int computeMode;
    int clockInstructionRate;
    int arch[2];
    int concurrentKernels;
    int pciDomainID;
    int pciBusID;
    int pciDeviceID;
    size_t maxSharedMemoryPerMultiProcessor;
    int isMultiGpuBoard;
    int canMapHostMemory;
    char gcnArchName[256];
    int integrated;
    int cooperativeLaunch;
    int cooperativeMultiDeviceLaunch;
    size_t sharedMemPerBlockOptin;
    int pageableMemoryAccess;
    int pageableMemoryAccessUsesHostPageTables;
    int directMode;
    char reserved[512];
};

enum hipDeviceAttribute_t {
    hipDeviceAttributeMaxThreadsPerBlock = 1,
    hipDeviceAttributeMaxBlockDimX = 2,
    hipDeviceAttributeMaxBlockDimY = 3,
    hipDeviceAttributeMaxBlockDimZ = 4,
    hipDeviceAttributeMaxGridDimX = 5,
    hipDeviceAttributeMaxGridDimY = 6,
    hipDeviceAttributeMaxGridDimZ = 7,
    hipDeviceAttributeMaxSharedMemoryPerBlock = 8,
    hipDeviceAttributeTotalConstantMemory = 9,
    hipDeviceAttributeWarpSize = 10,
    hipDeviceAttributeMaxRegistersPerBlock = 12,
    hipDeviceAttributeClockRate = 13,
    hipDeviceAttributeMemoryClockRate = 14,
    hipDeviceAttributeMemoryBusWidth = 15,
    hipDeviceAttributeMultiprocessorCount = 16,
    hipDeviceAttributeComputeMode = 20,
    hipDeviceAttributeL2CacheSize = 21,
    hipDeviceAttributeMaxThreadsPerMultiProcessor = 22,
    hipDeviceAttributeComputeCapabilityMajor = 23,
    hipDeviceAttributeComputeCapabilityMinor = 24,
    hipDeviceAttributeConcurrentKernels = 31,
    hipDeviceAttributePciBusId = 33,
    hipDeviceAttributePciDeviceId = 34,
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor = 81,
    hipDeviceAttributeIsMultiGpuBoard = 83,
    hipDeviceAttributeIntegrated = 84,
    hipDeviceAttributeCooperativeLaunch = 85,
    hipDeviceAttributeCooperativeMultiDeviceLaunch = 86,
};

enum hipJitOption {
    hipJitOptionMaxRegisters = 0,
    hipJitOptionThreadsPerBlock,
    hipJitOptionWallTime,
    hipJitOptionInfoLogBuffer,
    hipJitOptionInfoLogBufferSizeBytes,
    hipJitOptionErrorLogBuffer,
    hipJitOptionErrorLogBufferSizeBytes,
    hipJitOptionOptimizationLevel,
    hipJitOptionTargetFromContext,
    hipJitOptionTarget,
    hipJitOptionFallbackStrategy,
    hipJitOptionGenerateDebugInfo,
    hipJitOptionLogVerbose,
    hipJitOptionGenerateLineInfo,
    hipJitOptionCacheMode,
    hipJitOptionSm3xOpt,
    hipJitOptionFastCompile,
    hipJitOptionNumOptions
};

// Struct layout used by Clang/HIP for fat binaries
struct FatBinaryHeader {
    uint32_t magic;
    uint32_t version;
    const void* binary;
    uint64_t size;
};

// --- Hijack Manager ---

struct RegisteredKernel {
    std::string name;
    const void* hsaco_data;
    size_t hsaco_size;
    
    // Cached parsed kernel information
    mutable std::optional<HsacoModule> loaded_module;
    mutable const KernelInfo* cached_kernel_info = nullptr;
};

struct RegisteredVar {
    const void* host_var;
    std::string name;
    size_t size;
    bool is_constant;
    bool is_global;
    void* dev_ptr;
};

class HijackManager {
public:
    static HijackManager& Get() {
        static HijackManager instance;
        return instance;
    }

    bool IsGlobalEnabled() const { return global_enabled_; }

    bool ShouldSimulateKernel(const std::string& kernel_name) const {
        if (!global_enabled_) return false;
        if (filter_list_.empty()) return true;

        for (const auto& filter : filter_list_) {
            if (kernel_name.find(filter) != std::string::npos) {
                return true;
            }
        }
        return false;
    }

    bool parse_clang_offload_bundle(const void* bundle_data, const void*& co_data, size_t& co_size) {
        if (!bundle_data) return false;
        const char* ptr = reinterpret_cast<const char*>(bundle_data);

        // Check magic
        const std::string magic = "__CLANG_OFFLOAD_BUNDLE__";
        if (std::memcmp(ptr, magic.c_str(), magic.size()) != 0) {
            return false;
        }
        ptr += magic.size();

        uint64_t num_bundles = 0;
        std::memcpy(&num_bundles, ptr, sizeof(num_bundles));
        ptr += sizeof(num_bundles);

        // Sanity check to avoid memory issues with corrupt data
        if (num_bundles == 0 || num_bundles > 128) {
            return false;
        }

        for (uint64_t i = 0; i < num_bundles; ++i) {
            uint64_t offset = 0;
            uint64_t size = 0;
            uint64_t triple_len = 0;

            std::memcpy(&offset, ptr, sizeof(offset));
            ptr += sizeof(offset);
            std::memcpy(&size, ptr, sizeof(size));
            ptr += sizeof(size);
            std::memcpy(&triple_len, ptr, sizeof(triple_len));
            ptr += sizeof(triple_len);

            if (triple_len == 0 || triple_len > 1024) {
                return false;
            }

            std::string triple(ptr, triple_len);
            ptr += triple_len;

            // Check if this is the AMD GPU target for gfx90a (cdna2)
            if ((triple.find("hip-amdgcn-amd-amdhsa") != std::string::npos || 
                 triple.find("hipv4-amdgcn-amd-amdhsa") != std::string::npos) &&
                triple.find("gfx90a") != std::string::npos) {
                
                co_data = reinterpret_cast<const void*>(reinterpret_cast<const char*>(bundle_data) + offset);
                co_size = size;
                return true;
            }
        }

        return false;
    }

    void RegisterFatBinary(void** handle, const void* data) {
        auto* header = reinterpret_cast<const FatBinaryHeader*>(data);
        std::cout << "[gcnsim_hip] RegisterFatBinary: magic=0x" << std::hex << header->magic 
                  << std::dec << ", version=" << header->version 
                  << ", size=" << header->size << ", ptr=" << header->binary << std::endl;

        const void* hsaco_data = header->binary;
        size_t hsaco_size = header->size;

        if (header->magic == 0x48495046) { // "HIPF"
            const void* extracted_data = nullptr;
            size_t extracted_size = 0;
            if (parse_clang_offload_bundle(header->binary, extracted_data, extracted_size)) {
                std::cout << "[gcnsim_hip] Successfully parsed Clang Offload Bundle! Found gfx90a target: "
                          << "offset=" << (reinterpret_cast<const char*>(extracted_data) - reinterpret_cast<const char*>(header->binary))
                          << ", size=" << extracted_size << std::endl;
                hsaco_data = extracted_data;
                hsaco_size = extracted_size;
            } else {
                std::cout << "[gcnsim_hip] Warning: magic is HIPF but could not parse Clang Offload Bundle. Using raw pointer." << std::endl;
            }
        }

        last_fatbin_data_ = hsaco_data;
        last_fatbin_size_ = hsaco_size;
        fatbin_handles_[handle] = {hsaco_data, hsaco_size};
    }

    void RegisterFunction(
        void** fatBinHandle,
        const void* hostFun,
        const char* deviceName
    ) {
        const void* binary = nullptr;
        size_t size = 0;
        
        auto it = fatbin_handles_.find(fatBinHandle);
        if (it != fatbin_handles_.end()) {
            binary = it->second.first;
            size = it->second.second;
        } else {
            binary = last_fatbin_data_;
            size = last_fatbin_size_;
        }

        registered_kernels_[hostFun] = {
            .name = deviceName,
            .hsaco_data = binary,
            .hsaco_size = size,
            .loaded_module = std::nullopt,
            .cached_kernel_info = nullptr
        };
    }

    const RegisteredKernel* GetRegisteredKernel(const void* hostFun) const {
        auto it = registered_kernels_.find(hostFun);
        if (it != registered_kernels_.end()) {
            return &it->second;
        }
        return nullptr;
    }

    void RegisterVar(
        void** fatBinHandle,
        char* hostVar,
        char* deviceVar,
        const char* deviceName,
        int ext,
        size_t size,
        int constant,
        int global
    ) {
        (void)fatBinHandle; (void)deviceVar; (void)ext;
        void* dev_alloc = std::malloc(size);
        if (hostVar && dev_alloc) {
            std::memcpy(dev_alloc, hostVar, size);
        }
        RegisteredVar rv{
            .host_var = hostVar,
            .name = deviceName ? deviceName : "",
            .size = size,
            .is_constant = (constant != 0),
            .is_global = (global != 0),
            .dev_ptr = dev_alloc
        };
        registered_vars_[hostVar] = rv;
        if (deviceName && std::strlen(deviceName) > 0) {
            registered_vars_by_name_[deviceName] = hostVar;
        }
    }

    const RegisteredVar* FindRegisteredVar(const void* symbol) const {
        if (!symbol) return nullptr;
        auto it = registered_vars_.find(symbol);
        if (it != registered_vars_.end()) {
            return &it->second;
        }
        for (const auto& pair : registered_vars_) {
            if (pair.second.name == reinterpret_cast<const char*>(symbol)) {
                return &pair.second;
            }
        }
        return nullptr;
    }

    const KernelInfo* ResolveKernelInfo(const RegisteredKernel& reg) const {
        if (reg.cached_kernel_info) {
            return reg.cached_kernel_info;
        }
        if (!reg.loaded_module.has_value()) {
            std::string err;
            auto mod = load_hsaco_from_memory(
                reinterpret_cast<const uint8_t*>(reg.hsaco_data),
                reg.hsaco_size,
                err
            );
            if (!mod) {
                reg.loaded_module = std::nullopt;
                return nullptr;
            }
            reg.loaded_module = std::move(*mod);
        }
        
        for (const auto& k : reg.loaded_module->kernels) {
            if (k.name == reg.name) {
                reg.cached_kernel_info = &k;
                return &k;
            }
        }
        return nullptr;
    }

private:
    HijackManager() {
        const char* hijack_env = std::getenv("GCNSIM_HIJACK");
        if (hijack_env) {
            std::string env_str(hijack_env);
            global_enabled_ = (env_str == "1" || env_str == "true" || env_str == "on" || env_str == "ON");
        } else {
            global_enabled_ = true; // Enabled by default
        }

        const char* filter_env = std::getenv("GCNSIM_HIJACK_FILTER");
        if (filter_env) {
            std::string filter_str(filter_env);
            size_t pos = 0;
            while ((pos = filter_str.find(',')) != std::string::npos) {
                filter_list_.push_back(filter_str.substr(0, pos));
                filter_str.erase(0, pos + 1);
            }
            if (!filter_str.empty()) {
                filter_list_.push_back(filter_str);
            }
        }
    }

    bool global_enabled_ = true;
    std::vector<std::string> filter_list_;
    
    // Tracks handles to raw data
    const void* last_fatbin_data_ = nullptr;
    size_t last_fatbin_size_ = 0;
    std::unordered_map<void**, std::pair<const void*, size_t>> fatbin_handles_;
    
    // Mapping of host stub pointers to kernels
    std::unordered_map<const void*, RegisteredKernel> registered_kernels_;
    
    // Mapping of global/constant variables
    std::unordered_map<const void*, RegisteredVar> registered_vars_;
    std::unordered_map<std::string, const void*> registered_vars_by_name_;
};

} // namespace gcnsim

// --- Global macros and forward declarations for hijacked C APIs ---
#define HIP_LAUNCH_PARAM_END             ((void*)0x00)
#define HIP_LAUNCH_PARAM_BUFFER_POINTER  ((void*)0x01)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE     ((void*)0x02)

extern "C" {
gcnsim::hipError_t hipMalloc(void** devPtr, size_t size);
gcnsim::hipError_t hipFree(void* devPtr);
gcnsim::hipError_t hipMemcpy(void* dst, const void* src, size_t size, gcnsim::hipMemcpyKind kind);
gcnsim::hipError_t hipMemcpyAsync(void* dst, const void* src, size_t size, gcnsim::hipMemcpyKind kind, gcnsim::hipStream_t stream);

gcnsim::hipError_t hipStreamCreate(gcnsim::hipStream_t* stream);
gcnsim::hipError_t hipStreamCreateWithFlags(gcnsim::hipStream_t* stream, unsigned int flags);
gcnsim::hipError_t hipStreamDestroy(gcnsim::hipStream_t stream);
gcnsim::hipError_t hipStreamSynchronize(gcnsim::hipStream_t stream);
gcnsim::hipError_t hipDeviceSynchronize();
gcnsim::hipError_t hipGetLastError(void);

gcnsim::hipError_t hipGetDeviceCount(int* count);
gcnsim::hipError_t hipSetDevice(int device);
gcnsim::hipError_t hipGetDevice(int* device);
gcnsim::hipError_t hipGetDeviceProperties(gcnsim::hipDeviceProp_t* prop, int deviceId);
gcnsim::hipError_t hipGetDevicePropertiesR0600(gcnsim::hipDeviceProp_t* prop, int deviceId);
gcnsim::hipError_t hipGetDevicePropertiesR0000(gcnsim::hipDeviceProp_t* prop, int deviceId);
const char* hipGetErrorString(gcnsim::hipError_t hip_error);
gcnsim::hipError_t hipDeviceGetAttribute(int* pi, gcnsim::hipDeviceAttribute_t attr, int deviceId);
gcnsim::hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId);
gcnsim::hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags);
gcnsim::hipError_t hipDeviceDisablePeerAccess(int peerDeviceId);

gcnsim::hipError_t hipEventCreate(gcnsim::hipEvent_t* event);
gcnsim::hipError_t hipEventCreateWithFlags(gcnsim::hipEvent_t* event, unsigned int flags);
gcnsim::hipError_t hipEventRecord(gcnsim::hipEvent_t event, gcnsim::hipStream_t stream);
gcnsim::hipError_t hipEventSynchronize(gcnsim::hipEvent_t event);
gcnsim::hipError_t hipEventElapsedTime(float* ms, gcnsim::hipEvent_t start, gcnsim::hipEvent_t stop);
gcnsim::hipError_t hipEventDestroy(gcnsim::hipEvent_t event);
gcnsim::hipError_t hipEventQuery(gcnsim::hipEvent_t event);

gcnsim::hipError_t hipStreamWaitEvent(gcnsim::hipStream_t stream, gcnsim::hipEvent_t event, unsigned int flags);
gcnsim::hipError_t hipStreamQuery(gcnsim::hipStream_t stream);

gcnsim::hipError_t hipMemset(void* devPtr, int value, size_t count);
gcnsim::hipError_t hipMemsetAsync(void* devPtr, int value, size_t count, gcnsim::hipStream_t stream);
gcnsim::hipError_t hipMemsetD8(void* devPtr, unsigned char value, size_t count);
gcnsim::hipError_t hipMemsetD16(void* devPtr, unsigned short value, size_t count);
gcnsim::hipError_t hipMemsetD32(void* devPtr, int value, size_t count);
gcnsim::hipError_t hipMemsetD8Async(void* devPtr, unsigned char value, size_t count, gcnsim::hipStream_t stream);
gcnsim::hipError_t hipMemsetD16Async(void* devPtr, unsigned short value, size_t count, gcnsim::hipStream_t stream);
gcnsim::hipError_t hipMemsetD32Async(void* devPtr, int value, size_t count, gcnsim::hipStream_t stream);

gcnsim::hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags);
gcnsim::hipError_t hipHostFree(void* ptr);
gcnsim::hipError_t hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags);
gcnsim::hipError_t hipPointerGetAttributes(gcnsim::hipPointerAttribute_t* attributes, const void* ptr);

gcnsim::hipError_t hipMemcpyPeer(void* dst, int dstDeviceId, const void* src, int srcDeviceId, size_t size);
gcnsim::hipError_t hipMemcpyPeerAsync(void* dst, int dstDeviceId, const void* src, int srcDeviceId, size_t size, gcnsim::hipStream_t stream);
gcnsim::hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, gcnsim::hipMemcpyKind kind);
gcnsim::hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, gcnsim::hipMemcpyKind kind, gcnsim::hipStream_t stream);

gcnsim::hipError_t hipExtModuleLaunchKernel(
    gcnsim::hipFunction_t f,
    uint32_t globalWorkSizeX, uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
    uint32_t localWorkSizeX,  uint32_t localWorkSizeY,  uint32_t localWorkSizeZ,
    size_t sharedMemBytes, gcnsim::hipStream_t hStream,
    void** kernelParams, void** extra,
    gcnsim::hipEvent_t startEvent, gcnsim::hipEvent_t stopEvent,
    uint32_t flags
);
gcnsim::hipError_t hipModuleLoad(gcnsim::hipModule_t* module, const char* fname);
gcnsim::hipError_t hipModuleLoadData(gcnsim::hipModule_t* module, const void* image);
gcnsim::hipError_t hipModuleLoadDataEx(gcnsim::hipModule_t* module, const void* image, unsigned int numOptions, gcnsim::hipJitOption* options, void** optionValues);
gcnsim::hipError_t hipModuleGetFunction(gcnsim::hipFunction_t* function, gcnsim::hipModule_t module, const char* name);
gcnsim::hipError_t hipModuleUnload(gcnsim::hipModule_t module);

void __hipRegisterVar(
    void** fatBinHandle,
    char* hostVar,
    char* deviceVar,
    const char* deviceName,
    int ext,
    size_t size,
    int constant,
    int global
);
void __hipRegisterTexture(
    void** fatBinHandle,
    const void* hostVar,
    const void* deviceVar,
    const char* deviceName,
    int type,
    int norm,
    int ext
);
void __hipRegisterSurface(
    void** fatBinHandle,
    const void* hostVar,
    const void* deviceVar,
    const char* deviceName,
    int type,
    int ext
);

gcnsim::hipError_t hipGetSymbolAddress(void** devPtr, const void* symbol);
gcnsim::hipError_t hipGetSymbolSize(size_t* size, const void* symbol);
gcnsim::hipError_t hipMemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset, gcnsim::hipMemcpyKind kind);
gcnsim::hipError_t hipMemcpyToSymbolAsync(const void* symbol, const void* src, size_t count, size_t offset, gcnsim::hipMemcpyKind kind, gcnsim::hipStream_t stream);
gcnsim::hipError_t hipMemcpyFromSymbol(void* dst, const void* symbol, size_t count, size_t offset, gcnsim::hipMemcpyKind kind);
gcnsim::hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t count, size_t offset, gcnsim::hipMemcpyKind kind, gcnsim::hipStream_t stream);
}
