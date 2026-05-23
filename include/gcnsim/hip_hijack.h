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
};

} // namespace gcnsim

// --- Global macros and forward declarations for hijacked C APIs ---
#define HIP_LAUNCH_PARAM_END             ((void*)0x00)
#define HIP_LAUNCH_PARAM_BUFFER_POINTER  ((void*)0x01)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE     ((void*)0x02)

extern "C" {
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
gcnsim::hipError_t hipModuleGetFunction(gcnsim::hipFunction_t* function, gcnsim::hipModule_t module, const char* name);
gcnsim::hipError_t hipModuleUnload(gcnsim::hipModule_t module);
}
