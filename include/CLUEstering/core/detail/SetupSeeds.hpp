
#pragma once

#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/data_structures/internal/SeedArray.hpp"
#include <cstddef>
#include <optional>

namespace clue::detail {
  template <typename TQueue, typename TDev>
  inline void setup_seeds(TQueue& queue,
                          std::optional<internal::SeedArray<TDev>>& seeds,
                          std::size_t seed_candidates) {
    if (!seeds.has_value() || seeds->capacity() < seed_candidates) {
      seeds = internal::SeedArray<TDev>(queue, seed_candidates);
    } else {
      seeds->reset(queue);
    }
    alpaka::onHost::wait(queue);
  }

}  // namespace clue::detail
