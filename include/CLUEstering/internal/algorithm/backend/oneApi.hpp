#pragma once

#include <alpaka/alpaka.hpp>
#include "CLUEstering/internal/algorithm/backend/AlgorithmDispatch.hpp"

#if ALPAKA_LANG_SYCL && ALPAKA_LANG_ONEAPI

  #include <oneapi/dpl/algorithm>
  #include <oneapi/dpl/execution>
  #include <oneapi/dpl/numeric>
  #include <utility>

namespace clue::internal::algorithm {

  template <typename Api>
  struct OneDplDefaultPolicy;

  template <>
  struct OneDplDefaultPolicy<alpaka::api::OneApi>
  {
    static auto get() { return oneapi::dpl::execution::dpcpp_default; }
  };

template <>
struct AlgorithmDispatch::Op<alpaka::api::OneApi>
{
  // ---------------- count_if ----------------
  template <typename Api, typename It, typename Pred>
  auto count_if(Api, It first, It last, Pred pred) const
  {
    auto policy = OneDplDefaultPolicy<Api>::get();
    return oneapi::dpl::count_if(policy, first, last, pred);
  }

  template <typename Api, typename Policy, typename It, typename Pred>
  auto count_if(Api, Policy&& policy, It first, It last, Pred pred) const
  {
    return oneapi::dpl::count_if(std::forward<Policy>(policy), first, last, pred);
  }

  // ---------------- sort ----------------
  template <typename Api, typename RAIt>
  void sort(Api, RAIt first, RAIt last) const
  {
    auto policy = OneDplDefaultPolicy<Api>::get();
    oneapi::dpl::sort(policy, first, last);
  }

  template <typename Api, typename Policy, typename RAIt>
  void sort(Api, Policy&& policy, RAIt first, RAIt last) const
  {
    oneapi::dpl::sort(std::forward<Policy>(policy), first, last);
  }

  template <typename Api, typename RAIt, typename Comp>
  void sort(Api, RAIt first, RAIt last, Comp comp) const
  {
    auto policy = OneDplDefaultPolicy<Api>::get();
    oneapi::dpl::sort(policy, first, last, comp);
  }

  template <typename Api, typename Policy, typename RAIt, typename Comp>
  void sort(Api, Policy&& policy, RAIt first, RAIt last, Comp comp) const
  {
    oneapi::dpl::sort(std::forward<Policy>(policy), first, last, comp);
  }

  // ---------------- extrema: min_element ----------------
  template <typename Api, typename FwdIt>
  auto min_element(Api, FwdIt first, FwdIt last) const
  {
    auto policy = OneDplDefaultPolicy<Api>::get();
    return oneapi::dpl::min_element(policy, first, last);
  }

  template <typename Api, typename Policy, typename FwdIt>
  auto min_element(Api, Policy&& policy, FwdIt first, FwdIt last) const
  {
    return oneapi::dpl::min_element(std::forward<Policy>(policy), first, last);
  }

  template <typename Api, typename FwdIt, typename Comp>
  auto min_element(Api, FwdIt first, FwdIt last, Comp comp) const
  {
    auto policy = OneDplDefaultPolicy<Api>::get();
    return oneapi::dpl::min_element(policy, first, last, comp);
  }

  template <typename Api, typename Policy, typename FwdIt, typename Comp>
  auto min_element(Api, Policy&& policy, FwdIt first, FwdIt last, Comp comp) const
  {
    return oneapi::dpl::min_element(std::forward<Policy>(policy), first, last, comp);
  }

  // ---------------- extrema: max_element ----------------
  template <typename Api, typename FwdIt>
  auto max_element(Api, FwdIt first, FwdIt last) const
  {
    auto policy = OneDplDefaultPolicy<Api>::get();
    return oneapi::dpl::max_element(policy, first, last);
  }

  template <typename Api, typename Policy, typename FwdIt>
  auto max_element(Api, Policy&& policy, FwdIt first, FwdIt last) const
  {
    return oneapi::dpl::max_element(std::forward<Policy>(policy), first, last);
  }

  template <typename Api, typename FwdIt, typename Comp>
  auto max_element(Api, FwdIt first, FwdIt last, Comp comp) const
  {
    auto policy = OneDplDefaultPolicy<Api>::get();
    return oneapi::dpl::max_element(policy, first, last, comp);
  }

  template <typename Api, typename Policy, typename FwdIt, typename Comp>
  auto max_element(Api, Policy&& policy, FwdIt first, FwdIt last, Comp comp) const
  {
    return oneapi::dpl::max_element(std::forward<Policy>(policy), first, last, comp);
  }

  // ---------------- extrema: minmax_element ----------------
  template <typename Api, typename FwdIt>
  auto minmax_element(Api, FwdIt first, FwdIt last) const
  {
    auto policy = OneDplDefaultPolicy<Api>::get();
    return oneapi::dpl::minmax_element(policy, first, last);
  }

  template <typename Api, typename Policy, typename FwdIt>
  auto minmax_element(Api, Policy&& policy, FwdIt first, FwdIt last) const
  {
    return oneapi::dpl::minmax_element(std::forward<Policy>(policy), first, last);
  }

  template <typename Api, typename FwdIt, typename Comp>
  auto minmax_element(Api, FwdIt first, FwdIt last, Comp comp) const
  {
    auto policy = OneDplDefaultPolicy<Api>::get();
    return oneapi::dpl::minmax_element(policy, first, last, comp);
  }

  template <typename Api, typename Policy, typename FwdIt, typename Comp>
  auto minmax_element(Api, Policy&& policy, FwdIt first, FwdIt last, Comp comp) const
  {
    return oneapi::dpl::minmax_element(std::forward<Policy>(policy), first, last, comp);
  }

  // ---------------- reduce ----------------
  template <typename Api, typename It>
  auto reduce(Api, It first, It last) const
  {
    auto policy = OneDplDefaultPolicy<Api>::get();
    return oneapi::dpl::reduce(policy, first, last);
  }

  template <typename Api, typename Policy, typename It>
  auto reduce(Api, Policy&& policy, It first, It last) const
  {
    return oneapi::dpl::reduce(std::forward<Policy>(policy), first, last);
  }

  template <typename Api, typename It, typename T>
  T reduce(Api, It first, It last, T init) const
  {
    auto policy = OneDplDefaultPolicy<Api>::get();
    return oneapi::dpl::reduce(policy, first, last, init);
  }

  template <typename Api, typename Policy, typename It, typename T>
  T reduce(Api, Policy&& policy, It first, It last, T init) const
  {
    return oneapi::dpl::reduce(std::forward<Policy>(policy), first, last, init);
  }

  template <typename Api, typename It, typename T, typename BinOp>
  T reduce(Api, It first, It last, T init, BinOp op) const
  {
    auto policy = OneDplDefaultPolicy<Api>::get();
    return oneapi::dpl::reduce(policy, first, last, init, op);
  }

  template <typename Api, typename Policy, typename It, typename T, typename BinOp>
  T reduce(Api, Policy&& policy, It first, It last, T init, BinOp op) const
  {
    return oneapi::dpl::reduce(std::forward<Policy>(policy), first, last, init, op);
  }
};

} // namespace clue::internal::algorithm
#endif