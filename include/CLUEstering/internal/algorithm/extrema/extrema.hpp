#pragma once

#include <alpaka/alpaka.hpp>
#include <utility>

#include "CLUEstering/internal/algorithm/backend/AlgoDispatchTag.hpp"
#include "CLUEstering/internal/algorithm/backend/backendImp.hpp"

namespace clue::internal::algorithm {

  // ---------------- min_element ----------------
  template <typename ExecutionPolicy, typename ForwardIterator>
  ALPAKA_FN_HOST inline constexpr auto min_element(
      auto&& anyWithApi,
      ExecutionPolicy&& policy,
      ForwardIterator first,
      ForwardIterator last)
  {
    using Api = ALPAKA_TYPEOF(alpaka::getApi(anyWithApi));
      using Tag = typename AlgoDispatchTag<Api>::type;

    return AlgorithmDispatch::Op<Tag>{}.min_element(
        Api{},
        std::forward<ExecutionPolicy>(policy),
        first,
        last);
  }

  template <typename ForwardIterator>
  ALPAKA_FN_HOST inline constexpr auto min_element(
      auto&& anyWithApi,
      ForwardIterator first,
      ForwardIterator last)
  {
    using Api = ALPAKA_TYPEOF(alpaka::getApi(anyWithApi));
      using Tag = typename AlgoDispatchTag<Api>::type;

    return AlgorithmDispatch::Op<Tag>{}.min_element(
        Api{},
        first,
        last);
  }

  template <typename ForwardIterator, typename BinaryPredicate>
  ALPAKA_FN_HOST inline constexpr auto min_element(
      auto&& anyWithApi,
      ForwardIterator first,
      ForwardIterator last,
      BinaryPredicate comp)
  {
    using Api = ALPAKA_TYPEOF(alpaka::getApi(anyWithApi));
      using Tag = typename AlgoDispatchTag<Api>::type;

    return AlgorithmDispatch::Op<Tag>{}.min_element(
        Api{},
        first,
        last,
        comp);
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename BinaryPredicate>
  ALPAKA_FN_HOST inline constexpr auto min_element(
      auto&& anyWithApi,
      ExecutionPolicy&& policy,
      ForwardIterator first,
      ForwardIterator last,
      BinaryPredicate comp)
  {
    using Api = ALPAKA_TYPEOF(alpaka::getApi(anyWithApi));
      using Tag = typename AlgoDispatchTag<Api>::type;

    return AlgorithmDispatch::Op<Tag>{}.min_element(
        Api{},
        std::forward<ExecutionPolicy>(policy),
        first,
        last,
        comp);
  }

  // ---------------- max_element ----------------
  template <typename ExecutionPolicy, typename ForwardIterator>
  ALPAKA_FN_HOST inline constexpr auto max_element(
      auto&& anyWithApi,
      ExecutionPolicy&& policy,
      ForwardIterator first,
      ForwardIterator last)
  {
    using Api = ALPAKA_TYPEOF(alpaka::getApi(anyWithApi));
      using Tag = typename AlgoDispatchTag<Api>::type;

    return AlgorithmDispatch::Op<Tag>{}.max_element(
        Api{},
        std::forward<ExecutionPolicy>(policy),
        first,
        last);
  }

  template <typename ForwardIterator>
  ALPAKA_FN_HOST inline constexpr auto max_element(
      auto&& anyWithApi,
      ForwardIterator first,
      ForwardIterator last)
  {
    using Api = ALPAKA_TYPEOF(alpaka::getApi(anyWithApi));
      using Tag = typename AlgoDispatchTag<Api>::type;

    return AlgorithmDispatch::Op<Tag>{}.max_element(
        Api{},
        first,
        last);
  }

  template <typename ForwardIterator, typename BinaryPredicate>
  ALPAKA_FN_HOST inline constexpr auto max_element(
      auto&& anyWithApi,
      ForwardIterator first,
      ForwardIterator last,
      BinaryPredicate comp)
  {
    using Api = ALPAKA_TYPEOF(alpaka::getApi(anyWithApi));
    using Tag = typename AlgoDispatchTag<Api>::type;

    return AlgorithmDispatch::Op<Tag>{}.max_element(
        Api{},
        first,
        last,
        comp);
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename BinaryPredicate>
  ALPAKA_FN_HOST inline constexpr auto max_element(
      auto&& anyWithApi,
      ExecutionPolicy&& policy,
      ForwardIterator first,
      ForwardIterator last,
      BinaryPredicate comp)
  {
    using Api = ALPAKA_TYPEOF(alpaka::getApi(anyWithApi));
    using Tag = typename AlgoDispatchTag<Api>::type;

    return AlgorithmDispatch::Op<Tag>{}.max_element(
        Api{},
        std::forward<ExecutionPolicy>(policy),
        first,
        last,
        comp);
  }

  // ---------------- minmax_element ----------------
  template <typename ExecutionPolicy, typename ForwardIterator>
  ALPAKA_FN_HOST inline constexpr auto minmax_element(
      auto&& anyWithApi,
      ExecutionPolicy&& policy,
      ForwardIterator first,
      ForwardIterator last)
  {
    using Api = ALPAKA_TYPEOF(alpaka::getApi(anyWithApi));
    using Tag = typename AlgoDispatchTag<Api>::type;

    return AlgorithmDispatch::Op<Tag>{}.minmax_element(
        Api{},
        std::forward<ExecutionPolicy>(policy),
        first,
        last);
  }

  template <typename ForwardIterator>
  ALPAKA_FN_HOST inline constexpr auto minmax_element(
      auto&& anyWithApi,
      ForwardIterator first,
      ForwardIterator last)
  {
    using Api = ALPAKA_TYPEOF(alpaka::getApi(anyWithApi));
    using Tag = typename AlgoDispatchTag<Api>::type;

    return AlgorithmDispatch::Op<Tag>{}.minmax_element(
        Api{},
        first,
        last);
  }

  template <typename ForwardIterator, typename BinaryPredicate>
  ALPAKA_FN_HOST inline constexpr auto minmax_element(
      auto&& anyWithApi,
      ForwardIterator first,
      ForwardIterator last,
      BinaryPredicate comp)
  {
    using Api = ALPAKA_TYPEOF(alpaka::getApi(anyWithApi));
    using Tag = typename AlgoDispatchTag<Api>::type;

    return AlgorithmDispatch::Op<Tag>{}.minmax_element(
        Api{},
        first,
        last,
        comp);
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename BinaryPredicate>
  ALPAKA_FN_HOST inline constexpr auto minmax_element(
      auto&& anyWithApi,
      ExecutionPolicy&& policy,
      ForwardIterator first,
      ForwardIterator last,
      BinaryPredicate comp)
  {
    using Api = ALPAKA_TYPEOF(alpaka::getApi(anyWithApi));
    using Tag = typename AlgoDispatchTag<Api>::type;

    return AlgorithmDispatch::Op<Tag>{}.minmax_element(
        Api{},
        std::forward<ExecutionPolicy>(policy),
        first,
        last,
        comp);
  }

} // namespace clue::internal::algorithm
