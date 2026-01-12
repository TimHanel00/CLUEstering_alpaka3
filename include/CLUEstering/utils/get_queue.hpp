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
  template <typename T_Device>
  inline auto get_queue(T_Device const &device) {
    return device.makeQueue();
  }

}  // namespace clue
