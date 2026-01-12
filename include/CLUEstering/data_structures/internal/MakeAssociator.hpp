
#pragma once

#include "CLUEstering/data_structures/AssociationMap.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/algorithm/reduce/reduce.hpp"
#include "CLUEstering/internal/nostd/maximum.hpp"
#include <limits>
#include <span>

namespace clue::internal {

  template <clue::concepts::Queue TQueue>
  inline auto make_associator(TQueue& queue,
                              std::span<const int32_t> associations,
                              int32_t elements) {
    const auto bins = clue::internal::algorithm::reduce(associations.begin(),
                                                        associations.end(),
                                                        std::numeric_limits<int32_t>::lowest(),
                                                        clue::nostd::maximum<int32_t>{}) +
                      1;
    clue::AssociationMap<ALPAKA_TYPEOF(queue.getDevice())> map(elements, bins, queue);
    map.fill(elements, associations, queue);
    alpaka::onHost::wait(queue);
    return map;
  }
  template<typename T_Host>
  inline decltype(auto) make_associator(T_Host host_dev,std::span<const int32_t> associations, int32_t elements){
    const auto bins = std::reduce(associations.begin(),
                                  associations.end(),
                                  std::numeric_limits<int32_t>::lowest(),
                                  clue::nostd::maximum<int32_t>{}) +

                      1;
    clue::AssociationMap<T_Host> map(elements, bins);
    map.fill(associations);
    return map;
  }

}  // namespace clue::internal
