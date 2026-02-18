#pragma once

#include <alpaka/alpaka.hpp>
#include "CLUEstering/internal/algorithm/backend/AlgorithmDispatch.hpp"

#include <algorithm>
#include <execution>
#include <numeric>
#include <utility>

namespace clue::internal::algorithm {

template <>
struct AlgorithmDispatch::Op<alpaka::api::Host>
{
  // ---------------- count_if ----------------
  template <typename Api, typename It, typename Pred>
  auto count_if(Api, It first, It last, Pred pred) const
  {
    return std::count_if(first, last, pred);
  }

  template <typename Api, typename Policy, typename It, typename Pred>
  auto count_if(Api, Policy&& policy, It first, It last, Pred pred) const
  {
    return std::count_if(std::forward<Policy>(policy), first, last, pred);
  }

  // ---------------- sort ----------------
  template <typename Api, typename RAIt>
  void sort(Api, RAIt first, RAIt last) const
  {
    std::sort(first, last);
  }

  template <typename Api, typename Policy, typename RAIt>
  void sort(Api, Policy&& policy, RAIt first, RAIt last) const
  {
    std::sort(std::forward<Policy>(policy), first, last);
  }

  template <typename Api, typename RAIt, typename Comp>
  void sort(Api, RAIt first, RAIt last, Comp comp) const
  {

    std::sort(first, last, comp);
  }

  template <typename Api, typename Policy, typename RAIt, typename Comp>
  void sort(Api, Policy&& policy, RAIt first, RAIt last, Comp comp) const
  {
    std::sort(std::forward<Policy>(policy), first, last, comp);
  }

  // ---------------- extrema: min_element ----------------
  template <typename Api, typename FwdIt>
  auto min_element(Api, FwdIt first, FwdIt last) const
  {

    return std::min_element(first, last);
  }

  template <typename Api, typename Policy, typename FwdIt>
  auto min_element(Api, Policy&& policy, FwdIt first, FwdIt last) const
  {
    return std::min_element(std::forward<Policy>(policy), first, last);
  }

  template <typename Api, typename FwdIt, typename Comp>
  auto min_element(Api, FwdIt first, FwdIt last, Comp comp) const
  {

    return std::min_element(first, last, comp);
  }

  template <typename Api, typename Policy, typename FwdIt, typename Comp>
  auto min_element(Api, Policy&& policy, FwdIt first, FwdIt last, Comp comp) const
  {
    return std::min_element(std::forward<Policy>(policy), first, last, comp);
  }

  // ---------------- extrema: max_element ----------------
  template <typename Api, typename FwdIt>
  auto max_element(Api, FwdIt first, FwdIt last) const
  {
    return std::max_element(first, last);
  }

  template <typename Api, typename Policy, typename FwdIt>
  auto max_element(Api, Policy&& policy, FwdIt first, FwdIt last) const
  {
    return std::max_element(std::forward<Policy>(policy), first, last);
  }

  template <typename Api, typename FwdIt, typename Comp>
  auto max_element(Api, FwdIt first, FwdIt last, Comp comp) const
  {

    return std::max_element(first, last, comp);
  }

  template <typename Api, typename Policy, typename FwdIt, typename Comp>
  auto max_element(Api, Policy&& policy, FwdIt first, FwdIt last, Comp comp) const
  {
    return std::max_element(std::forward<Policy>(policy), first, last, comp);
  }

  // ---------------- extrema: minmax_element ----------------
  template <typename Api, typename FwdIt>
  auto minmax_element(Api, FwdIt first, FwdIt last) const
  {

    return std::minmax_element(first, last);
  }

  template <typename Api, typename Policy, typename FwdIt>
  auto minmax_element(Api, Policy&& policy, FwdIt first, FwdIt last) const
  {
    return std::minmax_element(std::forward<Policy>(policy), first, last);
  }

  template <typename Api, typename FwdIt, typename Comp>
  auto minmax_element(Api, FwdIt first, FwdIt last, Comp comp) const
  {
    return std::minmax_element(first, last, comp);
  }

  template <typename Api, typename Policy, typename FwdIt, typename Comp>
  auto minmax_element(Api, Policy&& policy, FwdIt first, FwdIt last, Comp comp) const
  {
    return std::minmax_element(std::forward<Policy>(policy), first, last, comp);
  }

  // ---------------- reduce ----------------
  template <typename Api, typename It>
  auto reduce(Api, It first, It last) const
  {

    return std::reduce(first, last);
  }

  template <typename Api, typename Policy, typename It>
  auto reduce(Api, Policy&& policy, It first, It last) const
  {
    return std::reduce(std::forward<Policy>(policy), first, last);
  }

  template <typename Api, typename It, typename T>
  T reduce(Api, It first, It last, T init) const
  {

    return std::reduce(first, last, init);
  }

  template <typename Api, typename Policy, typename It, typename T>
  T reduce(Api, Policy&& policy, It first, It last, T init) const
  {
    return std::reduce(std::forward<Policy>(policy), first, last, init);
  }

  template <typename Api, typename It, typename T, typename BinOp>
  T reduce(Api, It first, It last, T init, BinOp op) const
  {
    return std::reduce(first, last, init, op);
  }

  template <typename Api, typename Policy, typename It, typename T, typename BinOp>
  T reduce(Api, Policy&& policy, It first, It last, T init, BinOp op) const
  {
    return std::reduce(std::forward<Policy>(policy), first, last, init, op);
  }
};

} // namespace clue::internal::algorithm
