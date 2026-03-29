#pragma once
#include <cstdint>
#include <vector>

namespace gcnsim {

// Simple set-associative cache model for cycle simulation.
// Models gfx90a memory hierarchy:
//   vL1D: 16 KB, 4-way, 64B lines  (per-CU)
//   L2:    8 MB, 16-way, 128B lines (shared)
// Returns the latency tier for a given address.

enum class CacheHitLevel {
    L1,    // vL1D hit
    L2,    // L2 hit
    HBM    // cache miss — full HBM round-trip
};

struct CacheConfig {
    uint32_t l1_size_bytes  = 16384;  // 16 KB
    uint32_t l1_ways        = 4;
    uint32_t l1_line_bytes  = 64;
    uint32_t l2_size_bytes  = 8 * 1024 * 1024;  // 8 MB
    uint32_t l2_ways        = 16;
    uint32_t l2_line_bytes  = 128;
    uint32_t l1_hit_cycles  = 100;
    uint32_t l2_hit_cycles  = 200;
    // HBM miss latency comes from LaunchConfig::global_mem_latency
};

class CacheSet {
public:
    CacheSet() = default;
    explicit CacheSet(uint32_t ways) : tags_(ways, INVALID_TAG), lru_(ways) {
        for (uint32_t i = 0; i < ways; i++) lru_[i] = i;
    }

    // Returns true on hit, false on miss (and inserts/evicts on miss).
    bool access(uint64_t tag);

private:
    static constexpr uint64_t INVALID_TAG = ~0ULL;
    std::vector<uint64_t> tags_;
    std::vector<uint32_t> lru_;  // lru_[0] = MRU, lru_[back] = LRU
    void promote(uint32_t way);
};

class CacheLevel {
public:
    CacheLevel() = default;
    CacheLevel(uint32_t size_bytes, uint32_t ways, uint32_t line_bytes);

    // Returns true on hit.
    bool access(uint64_t addr);

    uint64_t hits()   const { return hits_; }
    uint64_t misses() const { return misses_; }

private:
    uint32_t line_shift_ = 0;
    uint32_t num_sets_ = 0;
    std::vector<CacheSet> sets_;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
};

class CacheModel {
public:
    CacheModel() = default;
    explicit CacheModel(const CacheConfig& cfg);

    // Probe L1 then L2. Returns the hit level.
    CacheHitLevel access(uint64_t addr);

    // Stats
    uint64_t l1_hits()   const { return l1_.hits(); }
    uint64_t l1_misses() const { return l1_.misses(); }
    uint64_t l2_hits()   const { return l2_.hits(); }
    uint64_t l2_misses() const { return l2_.misses(); }

private:
    CacheLevel l1_;
    CacheLevel l2_;
};

} // namespace gcnsim
