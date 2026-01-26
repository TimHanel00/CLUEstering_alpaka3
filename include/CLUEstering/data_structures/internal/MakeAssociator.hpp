
#pragma once
#include "CLUEstering/detail/concepts.hpp"
#include <limits>
#include <span>

namespace clue::internal {

  template <typename TQueue>
  inline auto make_associator(TQueue& queue,
                              std::span<const int32_t> associations,
                              int32_t elements) {
    alpaka::onHost::SharedBuffer compute_buffer_out =
        alpaka::onHost::allocHost<int32_t>(alpaka::Vec{1U});
    alpaka::onHost::allocLikeDeferred(queue, compute_buffer_out);
    alpaka::onHost::memset(queue, compute_buffer_out, std::numeric_limits<int32_t>::lowest());

    auto in_buf_d = alpaka::onHost::allocLikeDeferred(queue, associations);

    alpaka::onHost::memcpy(queue, in_buf_d, associations);
    const auto bins = alpaka::onHost::reduce(queue,
                                             DevicePool::exec(),
                                             std::numeric_limits<int32_t>::lowest(),
                                             compute_buffer_out,
                                             alpaka::math::max,
                                             in_buf_d) +
                      1;

    DevAssociationMap map(queue, elements, bins);
    map.fill(queue, elements, associations);
    alpaka::onHost::wait(queue);
    return map;
  }
  inline auto make_associator(std::span<const int32_t> associations, int32_t elements)
      -> decltype(auto) {
    const auto bins = *std::ranges::max_element(associations) + 1;
    HostAssociationMap map(elements, bins);
    map.fill(associations);
    return map;
  }

}  // namespace clue::internal
