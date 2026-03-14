#pragma once
#include <cstdint>
#include <cstring>

namespace gcnsim {

// Direct host memory dereference — safe for simulation where all pointers
// are host malloc'd. In a real implementation this would bounds-check.

inline uint32_t mem_load_u32(uint64_t addr) {
    uint32_t v;
    std::memcpy(&v, reinterpret_cast<const void*>(addr), sizeof(v));
    return v;
}

inline uint64_t mem_load_u64(uint64_t addr) {
    uint64_t v;
    std::memcpy(&v, reinterpret_cast<const void*>(addr), sizeof(v));
    return v;
}

inline void mem_store_u32(uint64_t addr, uint32_t val) {
    std::memcpy(reinterpret_cast<void*>(addr), &val, sizeof(val));
}

inline void mem_store_u64(uint64_t addr, uint64_t val) {
    std::memcpy(reinterpret_cast<void*>(addr), &val, sizeof(val));
}

inline float mem_load_f32(uint64_t addr) {
    float v;
    std::memcpy(&v, reinterpret_cast<const void*>(addr), sizeof(v));
    return v;
}

inline void mem_store_f32(uint64_t addr, float val) {
    std::memcpy(reinterpret_cast<void*>(addr), &val, sizeof(val));
}

} // namespace gcnsim
