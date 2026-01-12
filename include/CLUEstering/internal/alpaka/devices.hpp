
#pragma once

#include <cassert>
#include <vector>

#include <alpaka/alpaka.hpp>

namespace clue {
  // return the alpaka accelerator devices for the given platform
  template <typename T_DeviceSelector>
  auto devices(T_DeviceSelector const &selector) {
    std::vector<ALPAKA_TYPEOF(selector.makeDevice(0u))> devices;
    for (uint32_t devId=0;devId<selector.getDeviceCount();devId++) {
      devices.emplace_back(selector.makeDevice(devId));
    }
    return devices;
  }

}  // namespace clue
