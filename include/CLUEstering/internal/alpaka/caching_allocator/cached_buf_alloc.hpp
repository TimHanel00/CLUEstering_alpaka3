
#pragma once

#include <alpaka/alpaka.hpp>
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/alpaka/caching_allocator/get_device_caching_allocator.hpp"
#include "CLUEstering/internal/alpaka/caching_allocator/get_host_caching_allocator.hpp"

namespace clue {
  template <alpaka::concepts::Vector T>
  auto getVec(T&& extent) {
    return std::forward<T>(extent);
  }

  template <typename T>
  requires std::integral<std::remove_cvref_t<T>>
  auto getVec(T&& extent) { return alpaka::Vec{static_cast<std::remove_cvref_t<T>>(extent)}; }
  namespace traits {
    template <class...>
    inline constexpr bool always_false_v = false;
    //! The caching memory allocator trait.
    template <typename TElem, typename TDevice, typename TQueue>
    struct CachedBufAlloc {
      static_assert(always_false_v<TElem, TDevice, TQueue>,
                    "This device does not support a caching allocator");
    };
    //! The caching memory allocator implementation for the CPU device
    template <typename TElem, concepts::HostApi TDev, concepts::HostApi TQueue>
    struct CachedBufAlloc<TElem, TDev, TQueue> {
      ALPAKA_FN_HOST static auto allocCachedBuf(TDev const& dev, TQueue& queue, auto&& extent)
          -> auto{
        // non-cached host-only memory
        return alpaka::onHost::allocHost<TElem>(ALPAKA_FORWARD(extent));
      }
    };
    //! The caching memory allocator implementation for the pinned host memory (requires the queue to be based on a non-host api)
    template <typename TElem, concepts::HostApi TDev, concepts::NonHostApi TQueue>
    struct CachedBufAlloc<TElem, TDev, TQueue> {
      ALPAKA_FN_HOST static auto allocCachedBuf(TDev const& dev, TQueue& queue, auto&& extent) {
        auto& allocator = getHostCachingAllocator(queue);
        alpaka::Vec const extents_vec = extent;
        size_t size = extents_vec.product();
        size_t sizeBytes = size * sizeof(TElem);
        void* memPtr = allocator.allocate(sizeBytes, queue);
        // use a custom deleter to return the buffer to the CachingAllocator
        auto deleter = [alloc = &allocator](TElem* ptr) { alloc->free(ptr); };
        auto pitchMd = alpaka::calculatePitchesFromExtents<TElem>(extents_vec);
        return alpaka::onHost::SharedBuffer{alpaka::api::host,
                                            reinterpret_cast<TElem*>(memPtr),
                                            ALPAKA_FORWARD(extent),
                                            std::move(pitchMd),
                                            std::move(deleter)};
      };
    };

    //! The caching memory allocator implementation for the device
    template <typename TElem, concepts::NonHostApi T_Device, concepts::NonHostApi TQueue>
    struct CachedBufAlloc<TElem, T_Device, TQueue> {
      ALPAKA_FN_HOST static auto allocCachedBuf(T_Device const& device,
                                                TQueue& queue,
                                                auto&& extent) {
        auto api = alpaka::getApi(device);  //hostApi
        auto& allocator = getDeviceCachingAllocator(device, queue);
        alpaka::Vec const extents_vec = extent;
        size_t size = extents_vec.product();
        size_t sizeBytes = size * sizeof(TElem);

        void* memPtr = allocator.allocate(sizeBytes, queue);
        auto* ptr = static_cast<TElem*>(memPtr);
        // use a custom deleter to return the buffer to the CachingAllocator
        auto deleter = [alloc = &allocator, ptr] { alloc->free(ptr); };
        auto pitchMd = alpaka::calculatePitchesFromExtents<TElem>(extents_vec);
        return alpaka::onHost::SharedBuffer{
            api, ptr, ALPAKA_FORWARD(extents_vec), std::move(pitchMd), deleter};
      };
    };

  }  // namespace traits
  template <typename TElem, typename TDev, concepts::Queue TQueue>
  ALPAKA_FN_HOST auto allocCachedBuf(TDev const& dev, TQueue& queue, auto const& extent) {
    return traits::CachedBufAlloc<TElem, TDev, TQueue>::allocCachedBuf(dev, queue, extent);
  }

}  // namespace clue
