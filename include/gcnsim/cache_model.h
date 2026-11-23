#pragma once
#include <cstdint>
#include <mutex>
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
    L3,    // L3 Infinity Cache hit (CDNA3)
    HBM    // cache miss — full HBM round-trip
};

struct CacheConfig {
    uint32_t l1_size_bytes  = 16384;  // 16 KB (32 KB on CDNA3)
    uint32_t l1_ways        = 4;
    uint32_t l1_line_bytes  = 64;
    uint32_t l2_size_bytes  = 8 * 1024 * 1024;  // 8 MB (32 MB on CDNA3)
    uint32_t l2_ways        = 16;
    uint32_t l2_line_bytes  = 128;
    uint32_t l1_hit_cycles  = 100;
    uint32_t l2_hit_cycles  = 200;
    // L3 Infinity Cache (CDNA3)
    bool     enable_l3      = false;
    uint32_t l3_size_bytes  = 256 * 1024 * 1024; // 256 MB
    uint32_t l3_ways        = 16;
    uint32_t l3_line_bytes  = 128;
    uint32_t l3_hit_cycles  = 180;
    // Per-XCD L2 & finite bandwidth / queuing model (opt-in for CDNA3 / MI300X).
    // Public sources:
    //   - Topology & HBM3 peak BW (8 XCDs, 4 MB L2/XCD, 256 MB MALL, 5.3 TB/s HBM3):
    //     AMD Instinct MI300X Datasheet & AMD CDNA 3 Architecture Whitepaper (public).
    //   - Per-tier internal BW & unloaded latency defaults below are approximate
    //     defaults derived from public MI300X microbenchmark literature (not official
    //     AMD datasheet guarantees) and can be overridden in config/env.
    uint32_t num_xcds           = 1;
    bool     enable_bw_model    = false;
    double   l1_bytes_per_cycle = 128.0;   // public microbenchmark estimate (per CU)
    double   l2_bytes_per_cycle = 768.0;   // public microbenchmark estimate (per XCD, ~1.6 TB/s @ 2.1 GHz)
    double   l3_bytes_per_cycle = 4096.0;  // public microbenchmark estimate (shared MALL, ~8.6 TB/s @ 2.1 GHz)
    double   hbm_bytes_per_cycle = 2524.0; // MI300X public 5.3 TB/s HBM3 / 2.1 GHz boost
    uint32_t hbm_base_cycles    = 0;       // 0 = fallback to LaunchConfig::global_mem_latency
};

struct CacheAccessResult {
    CacheHitLevel level = CacheHitLevel::HBM;
    uint64_t completion_cycle = 0;
    uint32_t bytes = 0;
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
    explicit CacheModel(const CacheConfig& cfg, uint32_t num_cus = 1);

    // Probe per-CU L1 then shared/per-XCD L2 (and L3 if enabled). Returns the hit level.
    CacheHitLevel access(uint64_t addr, uint32_t cu_id = 0);

    // Timed probe with byte accounting and optional bandwidth/queuing delay.
    CacheAccessResult access_timed(uint64_t addr, uint32_t cu_id, uint64_t issue_cycle,
                                   uint32_t req_bytes, uint32_t hbm_latency_fallback);

    const CacheConfig& config() const { return cfg_; }

    // Stats (L1/L2 stats aggregated across all per-CU L1 / per-XCD L2 instances)
    uint64_t l1_hits()   const;
    uint64_t l1_misses() const;
    uint64_t l2_hits()   const;
    uint64_t l2_misses() const;
    uint64_t l3_hits()   const { std::lock_guard<std::mutex> lock(mutex_); return l3_.hits(); }
    uint64_t l3_misses() const { std::lock_guard<std::mutex> lock(mutex_); return l3_.misses(); }
    uint64_t l1_bytes()  const { std::lock_guard<std::mutex> lock(mutex_); return l1_bytes_; }
    uint64_t l2_bytes()  const { std::lock_guard<std::mutex> lock(mutex_); return l2_bytes_; }
    uint64_t l3_bytes()  const { std::lock_guard<std::mutex> lock(mutex_); return l3_bytes_; }
    uint64_t hbm_bytes() const { std::lock_guard<std::mutex> lock(mutex_); return hbm_bytes_; }

private:
    uint32_t xcd_for_cu(uint32_t cu_id) const;

    mutable std::mutex mutex_;
    CacheConfig cfg_;
    uint32_t cus_per_xcd_ = 1;
    std::vector<CacheLevel> l1_per_cu_;
    std::vector<CacheLevel> l2_per_xcd_;
    bool       enable_l3_ = false;
    CacheLevel l3_;
    std::vector<double> l1_busy_until_;
    std::vector<double> l2_busy_until_;
    double l3_busy_until_  = 0.0;
    double hbm_busy_until_ = 0.0;
    uint64_t l1_bytes_  = 0;
    uint64_t l2_bytes_  = 0;
    uint64_t l3_bytes_  = 0;
    uint64_t hbm_bytes_ = 0;
};

} // namespace gcnsim
