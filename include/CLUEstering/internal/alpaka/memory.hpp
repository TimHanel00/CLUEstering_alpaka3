
#pragma once

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "CLUEstering/internal/alpaka/caching_allocator/allocator_policy.hpp"
#include "CLUEstering/detail/concepts.hpp"

#include <CLUEstering/utils/get_queue.hpp>

namespace clue {
  //alpaka common
  using Vec1D = alpaka::Vec<uint32_t, 1>;

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

  template <typename T_Data, concepts::Queue T_Queue>
  auto make_device_buffer(T_Queue&& queue, alpaka::concepts::VectorOrScalar auto&& extent) {
    return clue::alloc<T_Data>(queue.getDevice(), ALPAKA_FORWARD(queue), extent);
  }
  template <typename T_Data, typename T_Device>
  requires(!concepts::Queue<T_Device>) auto make_device_buffer(
      T_Device&& device, alpaka::concepts::VectorOrScalar auto&& extent) {
    return clue::alloc<T_Data>(ALPAKA_FORWARD(device), clue::get_queue(device), extent);
  }

  template <typename T_Data, concepts::Queue TQueue>
  auto make_device_buffer(TQueue&& queue) {
    return clue::alloc<T_Data>(queue.getDevice(), ALPAKA_FORWARD(queue), alpaka::Vec{1U});
  }

  template <typename T_Data, typename T_Device>
  requires(!concepts::Queue<T_Device>) auto make_device_buffer(T_Device&& device) {
    return clue::alloc<T_Data>(ALPAKA_FORWARD(device), clue::get_queue(device), alpaka::Vec{1U});
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
  template <typename T, concepts::Queue T_Queue>
  requires concepts::NonHostApi<T_Queue>
  auto make_host_buffer(T_Queue& queue, alpaka::concepts::VectorOrScalar auto const extent) {
    return clue::alloc<T>(alpaka::api::host, queue, extent);
  }
  template <class T_Device, class T_Data>
  struct GetBufferType;

  // Host: use host buffer type
  template <concepts::HostApi T_Device, class T_Data>
  struct GetBufferType<T_Device, T_Data> {
    using Elem = std::remove_extent_t<T_Data>;
    using type = decltype(make_host_buffer<Elem>(std::declval<std::size_t>()));
  };

  // Non-host: use device buffer type
  template <concepts::NonHostApi T_Device, class T_Data>
  struct GetBufferType<T_Device, T_Data> {
    using Elem = std::remove_extent_t<T_Data>;
    using type =
        decltype(make_device_buffer<Elem>(std::declval<T_Device>(), std::declval<std::size_t>()));
  };

  template <class T_Device, class T_Data>
  using getBufferType = typename GetBufferType<T_Device, T_Data>::type;

}  // namespace clue
