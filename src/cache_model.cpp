#include "gcnsim/cache_model.h"
#include <algorithm>
#include <cmath>

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
    if (ways == 0 || line_bytes == 0 || size_bytes == 0) {
        num_sets_ = 0;
        line_shift_ = 0;
        return;
    }
    // Compute log2(line_bytes) for shift
    line_shift_ = 0;
    uint32_t v = line_bytes;
    while (v > 1) { v >>= 1; line_shift_++; }

    num_sets_ = size_bytes / (ways * line_bytes);
    sets_.resize(num_sets_, CacheSet(ways));
}

bool CacheLevel::access(uint64_t addr) {
    if (num_sets_ == 0) return false;
    uint64_t line_addr = addr >> line_shift_;
    uint32_t set_idx = (uint32_t)(line_addr % num_sets_);
    uint64_t tag = line_addr / num_sets_;
    bool hit = sets_[set_idx].access(tag);
    if (hit) hits_++; else misses_++;
    return hit;
}

// --- CacheModel ---

CacheModel::CacheModel(const CacheConfig& cfg, uint32_t num_cus)
    : cfg_(cfg)
    , cus_per_xcd_(std::max(1u, std::max(1u, num_cus) / std::max(1u, cfg.num_xcds)))
    , l1_per_cu_(std::max(1u, num_cus), CacheLevel(cfg.l1_size_bytes, cfg.l1_ways, cfg.l1_line_bytes))
    , l2_per_xcd_(std::max(1u, cfg.num_xcds), CacheLevel(cfg.l2_size_bytes, cfg.l2_ways, cfg.l2_line_bytes))
    , enable_l3_(cfg.enable_l3)
    , l3_(cfg.enable_l3 ? cfg.l3_size_bytes : 0, cfg.enable_l3 ? cfg.l3_ways : 1, cfg.enable_l3 ? cfg.l3_line_bytes : 64)
    , l1_busy_until_(std::max(1u, num_cus), 0.0)
    , l2_busy_until_(std::max(1u, cfg.num_xcds), 0.0)
{}

uint32_t CacheModel::xcd_for_cu(uint32_t cu_id) const {
    if (l2_per_xcd_.empty()) return 0;
    return (cu_id / cus_per_xcd_) % static_cast<uint32_t>(l2_per_xcd_.size());
}

CacheHitLevel CacheModel::access(uint64_t addr, uint32_t cu_id) {
    return access_timed(addr, cu_id, 0, cfg_.l1_line_bytes, 300).level;
}

CacheAccessResult CacheModel::access_timed(uint64_t addr, uint32_t cu_id, uint64_t issue_cycle,
                                           uint32_t req_bytes, uint32_t hbm_latency_fallback) {
    std::lock_guard<std::mutex> lock(mutex_);
    CacheAccessResult res;
    uint32_t cu_idx = l1_per_cu_.empty() ? 0 : (cu_id % static_cast<uint32_t>(l1_per_cu_.size()));
    uint32_t xcd_idx = xcd_for_cu(cu_id);
    uint32_t l1_req = std::max(1u, req_bytes);

    auto charge_bw = [](double& busy_until, double arrival, uint32_t bytes, double bw) -> double {
        double start = std::max(arrival, busy_until);
        double dur = (bw > 0.0) ? (static_cast<double>(bytes) / bw) : 0.0;
        double end = start + dur;
        busy_until = end;
        return end;
    };

    l1_bytes_ += l1_req;
    if (!l1_per_cu_.empty() && l1_per_cu_[cu_idx].access(addr)) {
        res.level = CacheHitLevel::L1;
        res.bytes = l1_req;
        if (cfg_.enable_bw_model) {
            double end = charge_bw(l1_busy_until_[cu_idx], static_cast<double>(issue_cycle),
                                   l1_req, cfg_.l1_bytes_per_cycle);
            res.completion_cycle = static_cast<uint64_t>(std::ceil(end)) + cfg_.l1_hit_cycles;
        } else {
            res.completion_cycle = issue_cycle + cfg_.l1_hit_cycles;
        }
        return res;
    }

    uint32_t l2_req = cfg_.l1_line_bytes;
    l2_bytes_ += l2_req;
    if (!l2_per_xcd_.empty() && l2_per_xcd_[xcd_idx].access(addr)) {
        res.level = CacheHitLevel::L2;
        res.bytes = l2_req;
        if (cfg_.enable_bw_model) {
            double end = charge_bw(l2_busy_until_[xcd_idx], static_cast<double>(issue_cycle),
                                   l2_req, cfg_.l2_bytes_per_cycle);
            res.completion_cycle = static_cast<uint64_t>(std::ceil(end)) + cfg_.l2_hit_cycles;
        } else {
            res.completion_cycle = issue_cycle + cfg_.l2_hit_cycles;
        }
        return res;
    }

    uint32_t l3_req = cfg_.l2_line_bytes;
    if (enable_l3_) {
        l3_bytes_ += l3_req;
        if (l3_.access(addr)) {
            res.level = CacheHitLevel::L3;
            res.bytes = l3_req;
            if (cfg_.enable_bw_model) {
                double end = charge_bw(l3_busy_until_, static_cast<double>(issue_cycle),
                                       l3_req, cfg_.l3_bytes_per_cycle);
                res.completion_cycle = static_cast<uint64_t>(std::ceil(end)) + cfg_.l3_hit_cycles;
            } else {
                res.completion_cycle = issue_cycle + cfg_.l3_hit_cycles;
            }
            return res;
        }
    }

    uint32_t hbm_req = enable_l3_ ? cfg_.l3_line_bytes : cfg_.l2_line_bytes;
    uint32_t hbm_lat = (cfg_.hbm_base_cycles > 0) ? cfg_.hbm_base_cycles : hbm_latency_fallback;
    hbm_bytes_ += hbm_req;
    res.level = CacheHitLevel::HBM;
    res.bytes = hbm_req;
    if (cfg_.enable_bw_model) {
        double end = charge_bw(hbm_busy_until_, static_cast<double>(issue_cycle),
                               hbm_req, cfg_.hbm_bytes_per_cycle);
        res.completion_cycle = static_cast<uint64_t>(std::ceil(end)) + hbm_lat;
    } else {
        res.completion_cycle = issue_cycle + hbm_lat;
    }
    return res;
}

uint64_t CacheModel::l1_hits() const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t total = 0;
    for (const auto& l1 : l1_per_cu_) total += l1.hits();
    return total;
}

uint64_t CacheModel::l1_misses() const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t total = 0;
    for (const auto& l1 : l1_per_cu_) total += l1.misses();
    return total;
}

uint64_t CacheModel::l2_hits() const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t total = 0;
    for (const auto& l2 : l2_per_xcd_) total += l2.hits();
    return total;
}

uint64_t CacheModel::l2_misses() const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t total = 0;
    for (const auto& l2 : l2_per_xcd_) total += l2.misses();
    return total;
}

} // namespace gcnsim
