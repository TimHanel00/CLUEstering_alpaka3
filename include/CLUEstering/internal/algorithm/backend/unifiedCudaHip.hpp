#pragma once

#pragma once

#include <alpaka/alpaka.hpp>
#include "CLUEstering/internal/algorithm/backend/AlgorithmDispatch.hpp"
#if (ALPAKA_LANG_CUDA && (ALPAKA_COMP_CLANG_CUDA || ALPAKA_COMP_NVCC))
  #define CUDA_ENABLED
#endif
#if ALPAKA_LANG_HIP
  #define HIP_ENABLED
#endif

// Enable CUDA specialization only when compiling with a CUDA toolchain.
// (Do NOT gate on __CUDA_ARCH__ here: Thrust calls are typically host code launching kernels.)
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)

  #include <thrust/count.h>
  #include <thrust/execution_policy.h>
  #include <thrust/extrema.h>
  #include <thrust/reduce.h>
  #include <thrust/sort.h>
namespace clue::internal::algorithm {
  template <typename Api>
  struct ThrustDefaultPolicy;
#if defined(CUDA_ENABLED)
  template <>
  struct ThrustDefaultPolicy<alpaka::api::Cuda>
  {
    static auto get() { return thrust::device; }
  };
#endif
#if defined(HIP_ENABLED)
  template <>
  struct ThrustDefaultPolicy<alpaka::api::Hip>
  {
    static auto get() { return thrust::hip::par; }
  };
#endif

template <>
struct AlgorithmDispatch::Op<UnifiedCudaHip>
{
  // ---------------- count_if ----------------
  template <typename Api, typename It, typename Pred>
  auto count_if(Api , It first, It last, Pred pred) const
  {
    auto policy=ThrustDefaultPolicy<Api>::get();
    return thrust::count_if(policy, first, last, pred);
  }

  template <typename Api,typename Policy, typename It, typename Pred>
  auto count_if(Api, Policy&& policy, It first, It last, Pred pred) const
  {
    return thrust::count_if(std::forward<Policy>(policy), first, last, pred);
  }

  // ---------------- sort ----------------
  template <typename Api, typename RAIt>
  void sort(Api, RAIt first, RAIt last) const
  {
    auto policy = ThrustDefaultPolicy<Api>::get();
    thrust::sort(policy, first, last);
  }

  template <typename Api, typename Policy, typename RAIt>
  void sort(Api, Policy&& policy, RAIt first, RAIt last) const
  {
    thrust::sort(std::forward<Policy>(policy), first, last);
  }

  template <typename Api, typename RAIt, typename Comp>
  void sort(Api, RAIt first, RAIt last, Comp comp) const
  {
    auto policy = ThrustDefaultPolicy<Api>::get();
    thrust::sort(policy, first, last, comp);
  }

  template <typename Api, typename Policy, typename RAIt, typename Comp>
  void sort(Api, Policy&& policy, RAIt first, RAIt last, Comp comp) const
  {
    thrust::sort(std::forward<Policy>(policy), first, last, comp);
  }

  // ---------------- extrema: min_element ----------------
  template <typename Api, typename FwdIt>
  auto min_element(Api, FwdIt first, FwdIt last) const
  {
    auto policy = ThrustDefaultPolicy<Api>::get();
    return thrust::min_element(policy, first, last);
  }

  template <typename Api, typename Policy, typename FwdIt>
  auto min_element(Api, Policy&& policy, FwdIt first, FwdIt last) const
  {
    return thrust::min_element(std::forward<Policy>(policy), first, last);
  }

  template <typename Api, typename FwdIt, typename Comp>
  auto min_element(Api, FwdIt first, FwdIt last, Comp comp) const
  {
    auto policy = ThrustDefaultPolicy<Api>::get();
    return thrust::min_element(policy, first, last, comp);
  }

  template <typename Api, typename Policy, typename FwdIt, typename Comp>
  auto min_element(Api, Policy&& policy, FwdIt first, FwdIt last, Comp comp) const
  {
    return thrust::min_element(std::forward<Policy>(policy), first, last, comp);
  }


  // ---------------- extrema: max_element ----------------
  template <typename Api, typename FwdIt>
  auto max_element(Api, FwdIt first, FwdIt last) const
  {
    auto policy = ThrustDefaultPolicy<Api>::get();
    return thrust::max_element(policy, first, last);
  }

  template <typename Api, typename Policy, typename FwdIt>
  auto max_element(Api, Policy&& policy, FwdIt first, FwdIt last) const
  {
    return thrust::max_element(std::forward<Policy>(policy), first, last);
  }

  template <typename Api, typename FwdIt, typename Comp>
  auto max_element(Api, FwdIt first, FwdIt last, Comp comp) const
  {
    auto policy = ThrustDefaultPolicy<Api>::get();
    return thrust::max_element(policy, first, last, comp);
  }

  template <typename Api, typename Policy, typename FwdIt, typename Comp>
  auto max_element(Api, Policy&& policy, FwdIt first, FwdIt last, Comp comp) const
  {
    return thrust::max_element(std::forward<Policy>(policy), first, last, comp);
  }

  // ---------------- extrema: minmax_element ----------------
  template <typename Api, typename FwdIt>
  auto minmax_element(Api, FwdIt first, FwdIt last) const
  {
    auto policy = ThrustDefaultPolicy<Api>::get();
    return thrust::minmax_element(policy, first, last);
  }

  template <typename Api, typename Policy, typename FwdIt>
  auto minmax_element(Api, Policy&& policy, FwdIt first, FwdIt last) const
  {
    return thrust::minmax_element(std::forward<Policy>(policy), first, last);
  }

  template <typename Api, typename FwdIt, typename Comp>
  auto minmax_element(Api, FwdIt first, FwdIt last, Comp comp) const
  {
    auto policy = ThrustDefaultPolicy<Api>::get();
    return thrust::minmax_element(policy, first, last, comp);
  }

  template <typename Api, typename Policy, typename FwdIt, typename Comp>
  auto minmax_element(Api, Policy&& policy, FwdIt first, FwdIt last, Comp comp) const
  {
    return thrust::minmax_element(std::forward<Policy>(policy), first, last, comp);
  }


  // ---------------- reduce ----------------
  template <typename Api, typename It>
  auto reduce(Api, It first, It last) const
  {
    auto policy = ThrustDefaultPolicy<Api>::get();
    return thrust::reduce(policy, first, last);
  }

  template <typename Api, typename Policy, typename It>
  auto reduce(Api, Policy&& policy, It first, It last) const
  {
    return thrust::reduce(std::forward<Policy>(policy), first, last);
  }

  template <typename Api, typename It, typename T>
  T reduce(Api, It first, It last, T init) const
  {
    auto policy = ThrustDefaultPolicy<Api>::get();
    return thrust::reduce(policy, first, last, init);
  }

  template <typename Api, typename Policy, typename It, typename T>
  T reduce(Api, Policy&& policy, It first, It last, T init) const
  {
    return thrust::reduce(std::forward<Policy>(policy), first, last, init);
  }

  template <typename Api, typename It, typename T, typename BinOp>
  T reduce(Api, It first, It last, T init, BinOp op) const
  {
    auto policy = ThrustDefaultPolicy<Api>::get();
    return thrust::reduce(policy, first, last, init, op);
  }

  template <typename Api, typename Policy, typename It, typename T, typename BinOp>
  T reduce(Api, Policy&& policy, It first, It last, T init, BinOp op) const
  {
    return thrust::reduce(std::forward<Policy>(policy), first, last, init, op);
  }
};

}
#endif
