#include <gtest/gtest.h>
#include "gcnsim/cache_model.h"

using namespace gcnsim;

TEST(CacheModel, L1HitOnRepeatedAccess) {
    CacheModel cache(CacheConfig{});
    // First access: cold miss (L1 miss, L2 miss -> HBM)
    EXPECT_EQ(cache.access(0x1000), CacheHitLevel::HBM);
    // Second access to same address: L1 hit
    EXPECT_EQ(cache.access(0x1000), CacheHitLevel::L1);
}

TEST(CacheModel, L1HitSameLine) {
    CacheModel cache(CacheConfig{});
    // Access byte 0 of a 64B line
    EXPECT_EQ(cache.access(0x1000), CacheHitLevel::HBM);
    // Access byte 32 of the same 64B line -> L1 hit
    EXPECT_EQ(cache.access(0x1020), CacheHitLevel::L1);
}

TEST(CacheModel, L2HitAfterL1Eviction) {
    // L1: 16 KB, 4-way, 64B lines -> 64 sets
    // Fill set 0 with 4 lines (4-way), then a 5th evicts from L1 but stays in L2
    CacheConfig cfg;
    CacheModel cache(cfg);
    uint32_t l1_sets = cfg.l1_size_bytes / (cfg.l1_ways * cfg.l1_line_bytes);  // 64

    // Fill 4 ways of set 0
    for (int i = 0; i < 4; i++) {
        uint64_t addr = (uint64_t)i * l1_sets * cfg.l1_line_bytes;
        cache.access(addr);  // cold miss
    }
    // Access a 5th address mapping to the same L1 set -> evicts first entry from L1
    uint64_t evicting_addr = (uint64_t)4 * l1_sets * cfg.l1_line_bytes;
    cache.access(evicting_addr);

    // Original address evicted from L1 but should be in L2
    uint64_t evicted_addr = 0;
    EXPECT_EQ(cache.access(evicted_addr), CacheHitLevel::L2);
}

TEST(CacheModel, SequentialAccessReuse) {
    CacheModel cache(CacheConfig{});
    // Access 64 distinct L1 lines (stride = 64B), each on a different L1 set.
    // First pass: cold misses (mix of HBM and L2 hits due to 128B L2 lines).
    for (int i = 0; i < 64; i++) {
        uint64_t addr = (uint64_t)i * 64;
        cache.access(addr);  // populate both L1 and L2
    }
    // Second pass: all should be L1 hits (64 lines, one per set, no evictions)
    for (int i = 0; i < 64; i++) {
        uint64_t addr = (uint64_t)i * 64;
        EXPECT_EQ(cache.access(addr), CacheHitLevel::L1);
    }
}

TEST(CacheModel, StatsAccumulate) {
    CacheModel cache(CacheConfig{});
    cache.access(0x1000);  // HBM miss (L1 miss + L2 miss)
    cache.access(0x1000);  // L1 hit
    cache.access(0x2000);  // HBM miss
    cache.access(0x2000);  // L1 hit

    EXPECT_EQ(cache.l1_hits(), 2u);
    EXPECT_EQ(cache.l1_misses(), 2u);
    EXPECT_EQ(cache.l2_misses(), 2u);
}

TEST(CacheModel, CDNA3_L3InfinityCache) {
    CacheConfig cfg;
    cfg.l1_size_bytes = 1024; // 1 KB
    cfg.l1_ways = 2;
    cfg.l1_line_bytes = 64;
    cfg.l2_size_bytes = 4096; // 4 KB
    cfg.l2_ways = 4;
    cfg.l2_line_bytes = 64;
    cfg.enable_l3 = true;
    cfg.l3_size_bytes = 65536; // 64 KB
    cfg.l3_ways = 8;
    cfg.l3_line_bytes = 64;

    CacheModel cache(cfg);

    // Cold miss -> HBM
    EXPECT_EQ(cache.access(0x1000), CacheHitLevel::HBM);
    // Repeated access -> L1 hit
    EXPECT_EQ(cache.access(0x1000), CacheHitLevel::L1);

    // Evict from L2: fill L2 set for address 0x1000 with 4 accesses (4-way L2)
    uint32_t l2_sets = cfg.l2_size_bytes / (cfg.l2_ways * cfg.l2_line_bytes);
    for (int i = 1; i <= 4; i++) {
        uint64_t addr = 0x1000 + (uint64_t)i * l2_sets * cfg.l2_line_bytes;
        cache.access(addr);
    }

    // 0x1000 evicted from L1 and L2, but preserved in L3!
    EXPECT_EQ(cache.access(0x1000), CacheHitLevel::L3);
    EXPECT_EQ(cache.l3_hits(), 1u);
}
