
#pragma once

#include <alpaka/alpaka.hpp>
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/alpaka/caching_allocator/get_device_caching_allocator.hpp"
#include "CLUEstering/internal/alpaka/caching_allocator/get_host_caching_allocator.hpp"

namespace clue {

  namespace traits {
    template<class...>
    inline constexpr bool always_false_v = false;
    //! The caching memory allocator trait.
    template <typename TElem,
              typename TDevice,typename TQueue>
    struct CachedBufAlloc {
      static_assert(always_false_v<TElem, TDevice, TQueue>,
                    "This device does not support a caching allocator");
    };
    //! The caching memory allocator implementation for the CPU device
    template <typename TElem,concepts::HostApi TDev,concepts::HostApi TQueue>
    struct CachedBufAlloc<TElem,TDev,TQueue> {
      ALPAKA_FN_HOST static auto allocCachedBuf(TDev const& dev,
                                                TQueue const & queue,
                                                auto &&extent)
          -> auto {
        // non-cached host-only memory
        return alpaka::onHost::allocHost<TElem>(queue, ALPAKA_FORWARD(extent));
      }
    };
    //! The caching memory allocator implementation for the pinned host memory (requires the queue to be based on a non-host api)
    template <typename TElem,concepts::HostApi TDev,concepts::NonHostApi TQueue>
    struct CachedBufAlloc<TElem,TDev,TQueue> {
      ALPAKA_FN_HOST static auto allocCachedBuf(TDev const& dev,
                                                TQueue const &queue,
                                                auto && extent){
        auto& allocator = getHostCachingAllocator(queue);
        size_t size = extent.product();
        size_t sizeBytes = size * sizeof(TElem);
        void* memPtr = allocator.allocate(sizeBytes, queue);
        // use a custom deleter to return the buffer to the CachingAllocator
        auto deleter = [alloc = &allocator](TElem* ptr) { alloc->free(ptr); };
        return alpaka::onHost::SharedBuffer{alpaka::api::host,reinterpret_cast<TElem*>(memPtr), ALPAKA_FORWARD(extent),alpaka::onHost::getPitches(ALPAKA_FORWARD(extent)),std::move(deleter)};
      };
    };
    //! The caching memory allocator implementation for the device
    template <typename TElem,
              concepts::NonHostApi T_Device,concepts::NonHostApi TQueue>
    struct CachedBufAlloc<TElem,T_Device,TQueue> {
      ALPAKA_FN_HOST static auto allocCachedBuf(T_Device const &device,
                                                TQueue const &queue,
                                                auto && extent){

        auto api=alpaka::getApi(std::declval<T_Device>()); //hostApi
        auto& allocator = getDeviceCachingAllocator(device,queue);
        size_t size = extent.product();
        size_t sizeBytes = size * sizeof(TElem);
        void* memPtr = allocator.allocate(sizeBytes, queue);
        // use a custom deleter to return the buffer to the CachingAllocator
        auto deleter = [alloc = &allocator](TElem* ptr) { alloc->free(ptr); };
        return alpaka::onHost::SharedBuffer{api,reinterpret_cast<TElem*>(memPtr), ALPAKA_FORWARD(extent),alpaka::onHost::getPitches(ALPAKA_FORWARD(extent)),std::move(deleter)};
      };
    };

  }  // namespace traits
  template <typename TElem, typename TDev, concepts::Queue TQueue>
  ALPAKA_FN_HOST auto allocCachedBuf(TDev const& dev,
                                     TQueue const &queue,
                                     auto const& extent) {
    return traits::CachedBufAlloc<TElem, TDev,TQueue>::allocCachedBuf(
        dev, queue, extent);
  }

}  // namespace clue
