#pragma once
#include <alpaka/alpaka.hpp>

namespace clue::internal::algorithm{
  struct UnifiedCudaHip;
  template <typename Api>
  struct AlgoDispatchTag { using type = Api; };
  template <>
  struct AlgoDispatchTag<alpaka::api::Cuda> { using type = UnifiedCudaHip; };

  template <>
  struct AlgoDispatchTag<alpaka::api::Hip> { using type = UnifiedCudaHip; };

  template <typename Api>
  using AlgoDispatchTag_t = typename AlgoDispatchTag<Api>::type;
}