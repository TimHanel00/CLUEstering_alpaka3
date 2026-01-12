
#pragma once

#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/data_structures/internal/Followers.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>

namespace clue::detail {

  template <typename TQueue>
  void setup_followers(TQueue& queue, auto& followers, int32_t n_points) {
    if (!followers.has_value()) {
      followers = ALPAKA_TYPEOF(followers)(n_points, queue);
    }

    if (!(followers->extents() >= n_points)) {
      followers->initialize(n_points, queue);
    } else {
      followers->reset(n_points);
    }
  }

}  // namespace clue::detail
