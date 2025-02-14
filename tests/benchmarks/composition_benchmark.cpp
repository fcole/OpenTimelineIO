// SPDX-License-Identifier: Apache-2.0
// Copyright Contributors to the OpenTimelineIO project

#define OPENTIMELINEIO_TEST
#include "opentimelineio/composition.h"
#include "opentimelineio/clip.h"
#include "opentimelineio/errorStatus.h"
#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include <memory>

namespace otio = opentimelineio::OPENTIMELINEIO_VERSION;

// Original algorithm implementation for benchmarking
int64_t bisect_right_original(
    const std::vector<otio::SerializableObject::Retainer<otio::Composable>>& seq,
    otio::RationalTime const& tgt,
    std::function<otio::RationalTime(otio::Composable*)> const& key_func,
    otio::ErrorStatus* error_status,
    std::optional<int64_t> lower_search_bound = 0,
    std::optional<int64_t> upper_search_bound = std::nullopt) {
    
    if (*lower_search_bound < 0) {
        if (error_status) {
            *error_status = otio::ErrorStatus(
                otio::ErrorStatus::INTERNAL_ERROR,
                "lower_search_bound must be non-negative");
        }
        return 0;
    }

    if (!upper_search_bound) {
        upper_search_bound = seq.size();
    }

    int64_t midpoint_index = 0;
    while (*lower_search_bound < *upper_search_bound) {
        midpoint_index = static_cast<int64_t>(
            std::floor((*lower_search_bound + *upper_search_bound) / 2.0));

        if (tgt < key_func(seq[midpoint_index].value)) {
            upper_search_bound = midpoint_index;
        }
        else {
            lower_search_bound = midpoint_index + 1;
        }
    }

    return *lower_search_bound;
}

// Optimization V2: Memory access and branch prediction optimization
int64_t bisect_right_optimized_v2(
    const std::vector<otio::SerializableObject::Retainer<otio::Composable>>& seq,
    otio::RationalTime const& tgt,
    std::function<otio::RationalTime(otio::Composable*)> const& key_func,
    otio::ErrorStatus* error_status,
    std::optional<int64_t> lower_search_bound = 0,
    std::optional<int64_t> upper_search_bound = std::nullopt) {
    
    if (*lower_search_bound < 0) {
        if (error_status) {
            *error_status = otio::ErrorStatus(
                otio::ErrorStatus::INTERNAL_ERROR,
                "lower_search_bound must be non-negative");
        }
        return 0;
    }

    int64_t left = *lower_search_bound;
    int64_t right = upper_search_bound ? *upper_search_bound : seq.size();
    
    // Pre-fetch the target value to avoid repeated access
    const auto& target_val = tgt;
    
    // Main search loop with minimal branching
    while (left < right) {
        const int64_t mid = left + ((right - left) >> 1);
        // Pre-fetch the comparison value
        const auto& mid_val = key_func(seq[mid].value);
        // Use arithmetic instead of branching where possible
        left += (mid_val <= target_val) * (mid + 1 - left);
        right = right - (mid_val > target_val) * (right - mid);
    }

    return left;
}

// Optimization V3: SIMD-friendly approach with unrolled loop
int64_t bisect_right_optimized_v3(
    const std::vector<otio::SerializableObject::Retainer<otio::Composable>>& seq,
    otio::RationalTime const& tgt,
    std::function<otio::RationalTime(otio::Composable*)> const& key_func,
    otio::ErrorStatus* error_status,
    std::optional<int64_t> lower_search_bound = 0,
    std::optional<int64_t> upper_search_bound = std::nullopt) {
    
    if (*lower_search_bound < 0) {
        if (error_status) {
            *error_status = otio::ErrorStatus(
                otio::ErrorStatus::INTERNAL_ERROR,
                "lower_search_bound must be non-negative");
        }
        return 0;
    }

    int64_t left = *lower_search_bound;
    int64_t right = upper_search_bound ? *upper_search_bound : seq.size();
    
    // Pre-fetch the target value
    const auto& target_val = tgt;
    
    // Main search loop unrolled for better instruction pipelining
    while (right - left > 4) {
        const int64_t range = right - left;
        const int64_t mid1 = left + (range >> 2);
        const int64_t mid2 = left + (range >> 1);
        const int64_t mid3 = right - (range >> 2);
        
        // Pre-fetch all comparison values
        const auto& val1 = key_func(seq[mid1].value);
        const auto& val2 = key_func(seq[mid2].value);
        const auto& val3 = key_func(seq[mid3].value);
        
        // Use branchless comparisons to update bounds
        if (target_val < val1) {
            right = mid1;
        } else if (target_val < val2) {
            left = mid1 + 1;
            right = mid2;
        } else if (target_val < val3) {
            left = mid2 + 1;
            right = mid3;
        } else {
            left = mid3 + 1;
        }
    }
    
    // Final cleanup with branchless arithmetic
    while (left < right) {
        const int64_t mid = left + ((right - left) >> 1);
        const auto& mid_val = key_func(seq[mid].value);
        left += (mid_val <= target_val) * (mid + 1 - left);
        right = right - (mid_val > target_val) * (right - mid);
    }

    return left;
}

// Optimization V4: Cache-line aware with prefetching hints
int64_t bisect_right_optimized_v4(
    const std::vector<otio::SerializableObject::Retainer<otio::Composable>>& seq,
    otio::RationalTime const& tgt,
    std::function<otio::RationalTime(otio::Composable*)> const& key_func,
    otio::ErrorStatus* error_status,
    std::optional<int64_t> lower_search_bound = 0,
    std::optional<int64_t> upper_search_bound = std::nullopt) {
    
    if (*lower_search_bound < 0) {
        if (error_status) {
            *error_status = otio::ErrorStatus(
                otio::ErrorStatus::INTERNAL_ERROR,
                "lower_search_bound must be non-negative");
        }
        return 0;
    }

    int64_t left = *lower_search_bound;
    int64_t right = upper_search_bound ? *upper_search_bound : seq.size();
    
    // Pre-fetch the target value
    const auto& target_val = tgt;
    
    // Cache line is typically 64 bytes, so prefetch next likely positions
    constexpr size_t CACHE_LINE_SIZE = 64;
    constexpr size_t ELEMENTS_PER_CACHE_LINE = CACHE_LINE_SIZE / sizeof(otio::SerializableObject::Retainer<otio::Composable>);
    
    while (left < right) {
        const int64_t mid = left + ((right - left) >> 1);
        
        // Prefetch the next likely cache lines
        if (right - left > ELEMENTS_PER_CACHE_LINE) {
            const int64_t next_mid_lower = mid - (mid - left) / 2;
            const int64_t next_mid_upper = mid + (right - mid) / 2;
            __builtin_prefetch(&seq[next_mid_lower]);
            __builtin_prefetch(&seq[next_mid_upper]);
        }
        
        const auto& mid_val = key_func(seq[mid].value);
        const bool is_less = mid_val <= target_val;
        left += is_less * (mid + 1 - left);
        right = right - (!is_less) * (right - mid);
    }

    return left;
}

// Helper class to access protected methods
namespace opentimelineio { namespace OPENTIMELINEIO_VERSION {

class CompositionBenchmark {
public:
    static int64_t bisect_right(
        Composition* comp,
        RationalTime const& tgt,
        std::function<RationalTime(Composable*)> const& key_func,
        ErrorStatus* error_status,
        std::optional<int64_t> lower_search_bound = 0,
        std::optional<int64_t> upper_search_bound = std::nullopt) {
        return comp->_bisect_right(tgt, key_func, error_status, lower_search_bound, upper_search_bound);
    }
    
    static int64_t bisect_left(
        Composition* comp,
        RationalTime const& tgt,
        std::function<RationalTime(Composable*)> const& key_func,
        ErrorStatus* error_status,
        std::optional<int64_t> lower_search_bound = 0,
        std::optional<int64_t> upper_search_bound = std::nullopt) {
        return comp->_bisect_left(tgt, key_func, error_status, lower_search_bound, upper_search_bound);
    }
};

}} // namespace opentimelineio::OPENTIMELINEIO_VERSION

// Helper to create test data
static otio::SerializableObject::Retainer<otio::Composition> create_test_composition(int n) {
    auto comp = new otio::Composition();
    std::vector<otio::Composable*> children;
    children.reserve(n);
    
    for (int i = 0; i < n; i++) {
        children.push_back(new otio::Clip());
    }
    
    otio::ErrorStatus error_status;
    comp->set_children(children, &error_status);
    return comp;
}

// Benchmark functions
static void BM_BisectRight_InPlace(benchmark::State& state) {
    const int n = state.range(0);
    auto comp = create_test_composition(n);
    otio::RationalTime target(n/2, 1);
    auto key_func = [](otio::Composable* c) { return otio::RationalTime(1, 1); };
    
    for (auto _ : state) {
        otio::ErrorStatus error_status;
        benchmark::DoNotOptimize(otio::CompositionBenchmark::bisect_right(comp.value, target, key_func, &error_status, 0));
    }
}

static void BM_BisectLeft_InPlace(benchmark::State& state) {
    const int n = state.range(0);
    auto comp = create_test_composition(n);
    otio::RationalTime target(n/2, 1);
    auto key_func = [](otio::Composable* c) { return otio::RationalTime(1, 1); };
    
    for (auto _ : state) {
        otio::ErrorStatus error_status;
        benchmark::DoNotOptimize(otio::CompositionBenchmark::bisect_left(comp.value, target, key_func, &error_status, 0));
    }
}

BENCHMARK(BM_BisectRight_InPlace)
    ->RangeMultiplier(2)
    ->Range(8, 8<<10);

BENCHMARK(BM_BisectLeft_InPlace)
    ->RangeMultiplier(2)
    ->Range(8, 8<<10);

BENCHMARK_MAIN(); 