
#pragma once

#include <alpaka/alpaka.hpp>
namespace clue {
  template<class T>
  using ApiOf = ALPAKA_TYPEOF(alpaka::getApi(std::declval<T>()));
  template<class T>
  using DevType = ALPAKA_TYPEOF(std::declval<T>().getDevice());
}
namespace clue::concepts {



  template<class T>
  concept HostApi =
  std::is_same_v<ApiOf<T>, alpaka::api::Host>||std::is_same_v<T,alpaka::api::Host>;
  template<class T>
  concept NonHostApi =
  !HostApi<T>;
  template <class>
  struct is_queue_type : std::false_type {};

  template <alpaka::concepts::Api Api,
            alpaka::concepts::DeviceKind DevKind,
            alpaka::concepts::QueueKind QKind>
  struct is_queue_type<alpaka::onHost::Queue<alpaka::onHost::Device<Api, DevKind>, QKind>>
      : std::true_type {};


  template <class T>
  concept Queue = is_queue_type<std::remove_cvref_t<T>>::value;
  template <typename T>
  concept Pointer = std::is_pointer_v<T>;

  template <typename T>
  concept Numeric = requires {
    std::is_arithmetic_v<T>;
    requires sizeof(T) <= 8;
  };

}  // namespace clue::concepts
