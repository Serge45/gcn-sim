#include "gcnsim/cache_model.h"
#include <algorithm>

namespace gcnsim {

// --- CacheSet ---

bool CacheSet::access(uint64_t tag) {
    for (uint32_t i = 0; i < tags_.size(); i++) {
        if (tags_[i] == tag) {
            promote(i);
            return true;  // hit
        }
    }
    // Miss — evict LRU (last in lru_ order)
    uint32_t victim = lru_.back();
    tags_[victim] = tag;
    promote(victim);
    return false;
}

void CacheSet::promote(uint32_t way) {
    // Move 'way' to MRU position (front of lru_)
    auto it = std::find(lru_.begin(), lru_.end(), way);
    if (it != lru_.end()) {
        lru_.erase(it);
        lru_.insert(lru_.begin(), way);
    }
}

// --- CacheLevel ---

CacheLevel::CacheLevel(uint32_t size_bytes, uint32_t ways, uint32_t line_bytes) {
    // Compute log2(line_bytes) for shift
    line_shift_ = 0;
    uint32_t v = line_bytes;
    while (v > 1) { v >>= 1; line_shift_++; }

    num_sets_ = size_bytes / (ways * line_bytes);
    sets_.resize(num_sets_, CacheSet(ways));
}

bool CacheLevel::access(uint64_t addr) {
    uint64_t line_addr = addr >> line_shift_;
    uint32_t set_idx = (uint32_t)(line_addr % num_sets_);
    uint64_t tag = line_addr / num_sets_;
    bool hit = sets_[set_idx].access(tag);
    if (hit) hits_++; else misses_++;
    return hit;
}

// --- CacheModel ---

CacheModel::CacheModel(const CacheConfig& cfg)
    : l1_(cfg.l1_size_bytes, cfg.l1_ways, cfg.l1_line_bytes)
    , l2_(cfg.l2_size_bytes, cfg.l2_ways, cfg.l2_line_bytes)
{}

CacheHitLevel CacheModel::access(uint64_t addr) {
    if (l1_.access(addr)) return CacheHitLevel::L1;
    if (l2_.access(addr)) return CacheHitLevel::L2;
    return CacheHitLevel::HBM;
}

} // namespace gcnsim
