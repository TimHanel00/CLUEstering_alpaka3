
#pragma once
#include <memory>
#include "CLUEstering/internal/alpaka/minMax.hpp"
#include "CLUEstering/internal/algorithm/algorithm.hpp"
#include <limits>
#include <span>

#include <stdexcept>
#include <CLUEstering/internal/algorithm/extrema/extrema.hpp>
namespace clue::internal {

  template <typename TQueue,typename T_Elem>
  inline auto make_associator(TQueue& queue,
                              std::span<const T_Elem> associations,
                              T_Elem elements) {
    if (elements == 0 || associations.empty()) {
      alpaka::onHost::wait(queue);
      std::cerr << "make_associator: throwing early (elements=" << elements
                  << ", size=" << associations.size() << ")\n";
      throw std::invalid_argument("make_associator: elements and associations must be non-zero");
    }
    auto it_max=algorithm::max_element(queue,associations.begin(),associations.end());
    T_Elem max_el{};
    auto ptr = std::to_address(it_max);
    auto view=alpaka::makeView(alpaka::api::host,&max_el,Vec1D{T_Elem{1}});
    auto dev_view=alpaka::makeView(queue,ptr,Vec1D{T_Elem{1}});
    alpaka::onHost::memcpy(queue, view, dev_view,Vec1D{T_Elem{1}});
    alpaka::onHost::wait(queue);
    auto device=queue.getDevice();
    DevAssociationMap map(device, elements, max_el + 1);
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
