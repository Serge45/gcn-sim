#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <mutex>
#include <sstream>
#include <iomanip>
#include <cstdlib>

namespace gcnsim {

class MemoryFaultException : public std::runtime_error {
public:
    uint64_t address;
    MemoryFaultException(uint64_t addr, const std::string& msg)
        : std::runtime_error(msg), address(addr) {}
};

struct MemoryRegion {
    uint64_t base;
    size_t   size;
    std::string tag;
};

class MemoryManager {
public:
    static MemoryManager& Get() {
        static MemoryManager instance;
        return instance;
    }

    void RegisterRegion(uint64_t base, size_t size, const std::string& tag = "") {
        std::lock_guard<std::mutex> lock(mutex_);
        regions_.push_back({base, size, tag});
    }

    void UnregisterRegion(uint64_t base) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = regions_.begin(); it != regions_.end(); ++it) {
            if (it->base == base) {
                regions_.erase(it);
                break;
            }
        }
    }

    bool CheckAccess(uint64_t addr, size_t size) const {
        if (addr < 4096) {
            std::ostringstream ss;
            ss << "GPU Memory Fault: NULL pointer access at 0x"
               << std::hex << addr << " (size: " << std::dec << size << " bytes)";
            throw MemoryFaultException(addr, ss.str());
        }

        if (strict_mode_) {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto& r : regions_) {
                if (addr >= r.base && (addr + size) <= (r.base + r.size)) {
                    return true;
                }
            }
            std::ostringstream ss;
            ss << "GPU Memory Fault: Out of bounds access at 0x"
               << std::hex << addr << " (size: " << std::dec << size << " bytes)";
            throw MemoryFaultException(addr, ss.str());
        }

        return true;
    }

    void SetStrictMode(bool enable) { strict_mode_ = enable; }
    bool IsStrictMode() const { return strict_mode_; }

private:
    MemoryManager() {
        const char* env = std::getenv("GCNSIM_STRICT_MEM");
        if (env && (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0)) {
            strict_mode_ = true;
        }
    }

    mutable std::mutex mutex_;
    std::vector<MemoryRegion> regions_;
    bool strict_mode_ = false;
};

inline uint32_t mem_load_u32(uint64_t addr) {
    MemoryManager::Get().CheckAccess(addr, sizeof(uint32_t));
    uint32_t v;
    std::memcpy(&v, reinterpret_cast<const void*>(addr), sizeof(v));
    return v;
}

inline uint64_t mem_load_u64(uint64_t addr) {
    MemoryManager::Get().CheckAccess(addr, sizeof(uint64_t));
    uint64_t v;
    std::memcpy(&v, reinterpret_cast<const void*>(addr), sizeof(v));
    return v;
}

inline void mem_store_u32(uint64_t addr, uint32_t val) {
    MemoryManager::Get().CheckAccess(addr, sizeof(uint32_t));
    std::memcpy(reinterpret_cast<void*>(addr), &val, sizeof(val));
}

inline void mem_store_u64(uint64_t addr, uint64_t val) {
    MemoryManager::Get().CheckAccess(addr, sizeof(uint64_t));
    std::memcpy(reinterpret_cast<void*>(addr), &val, sizeof(val));
}

inline float mem_load_f32(uint64_t addr) {
    MemoryManager::Get().CheckAccess(addr, sizeof(float));
    float v;
    std::memcpy(&v, reinterpret_cast<const void*>(addr), sizeof(v));
    return v;
}

inline void mem_store_f32(uint64_t addr, float val) {
    MemoryManager::Get().CheckAccess(addr, sizeof(float));
    std::memcpy(reinterpret_cast<void*>(addr), &val, sizeof(val));
}

} // namespace gcnsim
