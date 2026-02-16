
#pragma once

#include "CLUEstering/internal/alpaka/minMax.hpp"
#include <limits>
#include <span>
#include <stdexcept>
namespace clue::internal {

  template <typename TQueue,typename T_Elem>
  inline auto make_associator(TQueue& queue,
                              std::span<const T_Elem> associations,
                              T_Elem elements) {
    if (elements == 0 || associations.empty()) {
      throw std::invalid_argument("make_associator: elements and associations must be non-zero");
    }
    auto device=queue.getDevice();
    alpaka::onHost::SharedBuffer compute_buffer_out =
        alpaka::onHost::alloc<T_Elem>(device,alpaka::Vec{1U});
    alpaka::onHost::fill(queue, compute_buffer_out, std::numeric_limits<T_Elem>::lowest());
    auto in_buf_d = alpaka::onHost::alloc<T_Elem>(device, Vec1D{associations.size()}); //non-const input bufffer required
    auto host_view=alpaka::makeView(queue,associations.data(),Vec1D{associations.size()});
    alpaka::onHost::memcpy(queue, in_buf_d, host_view);
    alpaka::onHost::reduce(queue,
                                             DevicePool::exec(),
                                             std::numeric_limits<T_Elem>::lowest(),
                                             compute_buffer_out,
                                             simdMax(),
                                             in_buf_d);
    alpaka::onHost::SharedBuffer out_h =
    alpaka::onHost::allocHost<T_Elem>(alpaka::Vec{1U});
    alpaka::onHost::memcpy(queue,out_h,compute_buffer_out);

    DevAssociationMap map(device, elements, out_h[0u] + 1);
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
