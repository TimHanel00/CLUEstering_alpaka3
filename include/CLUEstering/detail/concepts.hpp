
#pragma once

#include <alpaka/alpaka.hpp>

namespace clue::concepts {



  template<alpaka::onHost::concepts::DeviceSpec T>
  consteval bool is_cpu_device_spec_v(const T&) {
    return std::is_same_v<T,alpaka::deviceKind::IntelGpu>||std::is_same_v<T,alpaka::deviceKind::AmdGpu>||std::is_same_v<T,alpaka::deviceKind::Cpu>;
  }
  template<alpaka::onHost::concepts::Device T>
  concept CpuDevice=is_cpu_device_spec_v(std::declval<T const &>().getDeviceKind());
  template<
    alpaka::concepts::Api Api,
    alpaka::concepts::DeviceKind DevKind,
    alpaka::concepts::QueueKind QKind>
  constexpr bool is_queue(alpaka::onHost::Queue<alpaka::onHost::Device<Api, DevKind>, QKind> const&) {
    return true;
  }

  template <class T>
  constexpr bool is_queue(T const&) { return false; }

  template <class T>
  concept Queue = is_queue(std::declval<T const&>());
  template <typename T>
  concept Pointer = std::is_pointer_v<T>;

  template <typename T>
  concept Numeric = requires {
    std::is_arithmetic_v<T>;
    requires sizeof(T) <= 8;
  };

}  // namespace clue::concepts
