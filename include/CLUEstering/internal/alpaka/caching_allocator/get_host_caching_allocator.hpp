
#pragma once

#include "CLUEstering/internal/alpaka/caching_allocator/allocator_config.hpp"
#include "CLUEstering/internal/alpaka/caching_allocator/caching_allocator.hpp"
#include "CLUEstering/internal/alpaka/devices.hpp"

namespace clue {

  template <typename TQueue>
  inline auto& getHostCachingAllocator(TQueue const& queue) {
    // thread safe initialisation of the host allocator
    static CachingAllocator<alpaka::api::Host, TQueue> allocator(
        DevicePool::getHost(),
        config::binGrowth,
        config::minBin,
        config::maxBin,
        config::maxCachedBytes,
        config::maxCachedFraction,
        false,   // reuseSameQueueAllocations
        false);  // debug

    // the public interface is thread safe
    return allocator;
  }

}  // namespace clue
