/// @file get_queue.hpp
/// @brief Provides functions to get an alpaka queue from a device index or a device object
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include "CLUEstering/detail/concepts.hpp"
#include <concepts>
#include <queue>
#include <alpaka/alpaka.hpp>

namespace clue {
  /// @brief Get an alpaka queue created from a corresponding device
  template <std::integral T_Id,alpaka::concepts::QueueKind T_Kind=alpaka::queueKind::NonBlocking>
  inline auto get_queue(T_Id Id,T_Kind kind=T_Kind{}) {
    static auto queue=DevicePool::deviceAt(Id).makeQueue(kind);
    return queue;
  }
  /// @brief Get an alpaka queue created from a corresponding device
  template <alpaka::onHost::concepts::Device T_Device>
  inline auto get_queue(T_Device const &device) {

    return get_queue(DevicePool::indexOf(device));
  }

}  // namespace clue
