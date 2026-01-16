
#pragma once

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "CLUEstering/internal/alpaka/caching_allocator/allocator_policy.hpp"
#include "CLUEstering/internal/alpaka/caching_allocator/cached_buf_alloc.hpp"
#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/internal/alpaka/devices.hpp"
#include "CLUEstering/detail/concepts.hpp"

#include <CLUEstering/utils/get_queue.hpp>

namespace clue {

  namespace internal::concepts {

    // bounded 1D array
    template <typename T>
    concept bounded_array =
        std::is_bounded_array_v<T> && not std::is_array_v<std::remove_extent_t<T>>;

    // unbounded 1D array
    template <typename T>
    concept unbounded_array =
        std::is_unbounded_array_v<T> && not std::is_array_v<std::remove_extent_t<T>>;

    template <typename T>
    concept scalar = not std::is_array_v<T>;

  }  // namespace internal::concepts

  using namespace alpaka_common;
  template<typename T_Data,concepts::Queue T_Queue>
  auto make_device_buffer(T_Queue && queue,alpaka::concepts::VectorOrScalar auto && extent) {
    return clue::alloc<T_Data>(queue.getDevice(),ALPAKA_FORWARD(queue),extent);
  }
  template<typename T_Data,typename T_Device>
  requires (!concepts::Queue<T_Device>)
  auto make_device_buffer(T_Device && device,alpaka::concepts::VectorOrScalar auto && extent) {

    return clue::alloc<T_Data>(ALPAKA_FORWARD(device),clue::get_queue(device),extent);
  }
  using namespace alpaka_common;
  template<typename T_Data,concepts::Queue TQueue>
  auto make_device_buffer(TQueue && queue) {
    return clue::alloc<T_Data>(queue.getDevice(),ALPAKA_FORWARD(queue),Vec1D{1U});
  }
  using namespace alpaka_common;
  template<typename T_Data,typename T_Device>
  requires (!concepts::Queue<T_Device>)
  auto make_device_buffer(T_Device && device) {
    return clue::alloc<T_Data>(ALPAKA_FORWARD(device),clue::get_queue(device),Vec1D{1U});
  }


  template <typename T>
  auto make_host_buffer(alpaka::concepts::VectorOrScalar auto const extent) {
    return alpaka::onHost::allocHost<T>(extent);
  }
  template <typename T>
  auto make_host_buffer() {
    return alpaka::onHost::allocHost<T>(1U);
  }
  //to create pinned host memory
  template <typename T,concepts::Queue T_Queue>
  requires concepts::NonHostApi<T_Queue>
  auto make_host_buffer(T_Queue &queue,alpaka::concepts::VectorOrScalar auto const extent) {
    return clue::alloc<T>(alpaka::api::host,queue,extent);
  }
  template<class T_Device, class T_Data>
  struct GetBufferType;

  // Host: use host buffer type
  template<concepts::HostApi T_Device, class T_Data>
  struct GetBufferType<T_Device, T_Data> {
    using Elem = std::remove_extent_t<T_Data>;
    using type = decltype(make_host_buffer<Elem>(std::declval<std::size_t>()));
  };

  // Non-host: use device buffer type
  template<concepts::NonHostApi T_Device, class T_Data>
  struct GetBufferType<T_Device, T_Data> {
    using Elem = std::remove_extent_t<T_Data>;
    using type = decltype(make_device_buffer<Elem>(
        std::declval<T_Device>(), std::declval<std::size_t>()));
  };

  template<class T_Device, class T_Data>
  using getBufferType = typename GetBufferType<T_Device, T_Data>::type;
  // alpaka3 memory allocations are implicitly synchronized therefore we dont need a queue

  // // potentially cached, pinned, scalar and 1-dimensional host buffers, associated to a work queue
  // // the memory is pinned according to the device associated to the queue
  //
  // template <internal::concepts::scalar T, concepts::Queue TQueue>
  // host_buffer<T> make_host_buffer(TQueue const& queue) {
  //   return alpaka::onHost::allocHost<T>(queue);
  // }
  //
  // template <internal::concepts::unbounded_array T, concepts::Queue TQueue>
  // host_buffer<T> make_host_buffer(TQueue const& queue, alpaka::concepts::Vector auto extent) {
  //             host, platform<TPlatform>(), Vec1D{extent});
  // }
  //
  // template <internal::concepts::bounded_array T, concepts::Queue TQueue>
  // host_buffer<T> make_host_buffer(TQueue const& queue) {
  //   if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Caching) {
  //     return allocCachedBuf<std::remove_extent_t<T>, Idx>(host, queue, Vec1D{std::extent_v<T>});
  //   } else {
  //     using TPlatform = alpaka::Platform<alpaka::Dev<TQueue>>;
  //     return alpaka::allocMappedBuf<std::remove_extent_t<T>, Idx>(
  //         host, platform<TPlatform>(), Vec1D{std::extent_v<T>});
  //   }
  // }

  /***
   a alpaka3 buffer is essentially a non-owning view, which manages the lifetime of the underlying data
   on the device we use a IMdSpan (which may be provided by passing a buffer to the kernel
   furthermore this functionality is basically providede by makeView
  ****/
  // template <typename T>
  // using host_view = typename detail::view_type<DevHost, T>::type;
  //
  // template <internal::concepts::scalar T>
  // host_view<T> make_host_view(T& data) {
  //   return alpaka::ViewPlainPtr<DevHost, T, Dim0D, Idx>(&data, host, Scalar{});
  // }
  //
  // template <internal::concepts::scalar T>
  // host_view<T[]> make_host_view(T* data, Extent extent) {
  //   return alpaka::ViewPlainPtr<DevHost, T, Dim1D, Idx>(data, host, Vec1D{extent});
  // }
  //
  // template <internal::concepts::unbounded_array T>
  // host_view<T> make_host_view(T& data, Extent extent) {
  //   return alpaka::ViewPlainPtr<DevHost, std::remove_extent_t<T>, Dim1D, Idx>(
  //       data, host, Vec1D{extent});
  // }
  //
  // template <internal::concepts::bounded_array T>
  // host_view<T> make_host_view(T& data) {
  //   return alpaka::ViewPlainPtr<DevHost, std::remove_extent_t<T>, Dim1D, Idx>(
  //       data, host, Vec1D{std::extent_v<T>});
  // }
  //
  // // scalar and 1-dimensional device buffers
  //
  // template <typename TDev, typename T>
  // using device_buffer = typename detail::buffer_type<TDev, T>::type;
  //
  // template <internal::concepts::scalar T, concepts::Queue TQueue>
  // device_buffer<alpaka::Dev<TQueue>, T> make_device_buffer(TQueue const& queue) {
  //   if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Caching) {
  //     return allocCachedBuf<T, Idx>(alpaka::getDev(queue), queue, Scalar{});
  //   }
  //   if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Asynchronous) {
  //     return alpaka::allocAsyncBuf<T, Idx>(queue, Scalar{});
  //   }
  //   if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Synchronous) {
  //     return alpaka::allocBuf<T, Idx>(alpaka::getDev(queue), Scalar{});
  //   }
  // }
  //
  // template <internal::concepts::unbounded_array T, concepts::Queue TQueue>
  // device_buffer<alpaka::Dev<TQueue>, T> make_device_buffer(TQueue const& queue, Extent extent) {
  //   if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Caching) {
  //     return allocCachedBuf<std::remove_extent_t<T>, Idx>(
  //         alpaka::getDev(queue), queue, Vec1D{extent});
  //   }
  //   if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Asynchronous) {
  //     return alpaka::allocAsyncBuf<std::remove_extent_t<T>, Idx>(queue, Vec1D{extent});
  //   }
  //   if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Synchronous) {
  //     return alpaka::allocBuf<std::remove_extent_t<T>, Idx>(alpaka::getDev(queue), Vec1D{extent});
  //   }
  // }
  //
  // template <internal::concepts::bounded_array T, concepts::Queue TQueue>
  // device_buffer<alpaka::Dev<TQueue>, T> make_device_buffer(TQueue const& queue) {
  //   if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Caching) {
  //     return allocCachedBuf<std::remove_extent_t<T>, Idx>(
  //         alpaka::getDev(queue), queue, Vec1D{std::extent_v<T>});
  //   }
  //   if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Asynchronous) {
  //     return alpaka::allocAsyncBuf<std::remove_extent_t<T>, Idx>(queue, Vec1D{std::extent_v<T>});
  //   }
  //   if constexpr (allocator_policy<alpaka::Dev<TQueue>> == AllocatorPolicy::Synchronous) {
  //     return alpaka::allocBuf<std::remove_extent_t<T>, Idx>(alpaka::getDev(queue),
  //                                                           Vec1D{std::extent_v<T>});
  //   }
  // }
  //
  // // scalar and 1-dimensional device views
  //
  // template <typename TDev, typename T>
  // using device_view = typename detail::view_type<TDev, T>::type;
  //
  // template <internal::concepts::scalar T, alpaka::onHost::concepts::Device TDev>
  // device_view<TDev, T> make_device_view(TDev const& device, T& data) {
  //   return alpaka::ViewPlainPtr<TDev, T, Dim0D, Idx>(&data, device, Scalar{});
  // }
  //
  // template <internal::concepts::scalar T, alpaka::onHost::concepts::Device TDev>
  // device_view<TDev, T[]> make_device_view(TDev const& device, T* data, Extent extent) {
  //   return alpaka::ViewPlainPtr<TDev, T, Dim1D, Idx>(data, device, Vec1D{extent});
  // }
  //
  // template <internal::concepts::unbounded_array T, alpaka::onHost::concepts::Device TDev>
  // device_view<TDev, T> make_device_view(TDev const& device, T& data, Extent extent) {
  //   return alpaka::ViewPlainPtr<TDev, std::remove_extent_t<T>, Dim1D, Idx>(
  //       data, device, Vec1D{extent});
  // }
  //
  // template <internal::concepts::bounded_array T, alpaka::onHost::concepts::Device TDev>
  // device_view<TDev, T> make_device_view(TDev const& device, T& data) {
  //   return alpaka::ViewPlainPtr<TDev, std::remove_extent_t<T>, Dim1D, Idx>(
  //       data, device, Vec1D{std::extent_v<T>});
  // }

}  // namespace clue
