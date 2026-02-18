#pragma once

#include <alpaka/alpaka.hpp>
#include <utility>

#include "CLUEstering/internal/algorithm/backend/AlgoDispatchTag.hpp"
#include "CLUEstering/internal/algorithm/backend/backendImp.hpp"

namespace clue::internal::algorithm {

  template <typename ExecutionPolicy, typename ForwardIterator>
  ALPAKA_FN_HOST constexpr auto reduce(
      auto&& anyWithApi,
      ExecutionPolicy&& policy,
      ForwardIterator first,
      ForwardIterator last)
  {
    using Api = ALPAKA_TYPEOF(alpaka::getApi(anyWithApi));
    using Tag = typename AlgoDispatchTag<Api>::type;

    return AlgorithmDispatch::Op<Tag>{}.reduce(
        Api{},
        std::forward<ExecutionPolicy>(policy),
        first,
        last);
  }
  template <typename ForwardIterator>
  ALPAKA_FN_HOST constexpr auto reduce(
      auto&& anyWithApi,
      ForwardIterator first,
      ForwardIterator last)
  {
    using Api = ALPAKA_TYPEOF(alpaka::getApi(anyWithApi));
    using Tag = typename AlgoDispatchTag<Api>::type;
    return AlgorithmDispatch::Op<Tag>{}.reduce(
        Api{},
        first,
        last);
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename T>
  ALPAKA_FN_HOST constexpr T reduce(
      auto&& anyWithApi,
      ExecutionPolicy&& policy,
      ForwardIterator first,
      ForwardIterator last,
      T init)
  {
    using Api = ALPAKA_TYPEOF(alpaka::getApi(anyWithApi));
    using Tag = typename AlgoDispatchTag<Api>::type;

    return AlgorithmDispatch::Op<Tag>{}.reduce(
        Api{},
        std::forward<ExecutionPolicy>(policy),
        first,
        last,
        init);
  }

  template <typename ForwardIterator, typename T>
  ALPAKA_FN_HOST constexpr T reduce(
      auto&& anyWithApi,
      ForwardIterator first,
      ForwardIterator last,
      T init)
  {
    using Api = ALPAKA_TYPEOF(alpaka::getApi(anyWithApi));
    using Tag = typename AlgoDispatchTag<Api>::type;

    return AlgorithmDispatch::Op<Tag>{}.reduce(
        Api{},
        first,
        last,
        init);
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename T, typename BinaryOperation>
  ALPAKA_FN_HOST constexpr T reduce(
      auto&& anyWithApi,
      ExecutionPolicy&& policy,
      ForwardIterator first,
      ForwardIterator last,
      T init,
      BinaryOperation op)
  {
    using Api = ALPAKA_TYPEOF(alpaka::getApi(anyWithApi));
    using Tag = typename AlgoDispatchTag<Api>::type;

    return AlgorithmDispatch::Op<Tag>{}.reduce(
        Api{},
        std::forward<ExecutionPolicy>(policy),
        first,
        last,
        init,
        op);
  }

  template <typename ForwardIterator, typename T, typename BinaryOperation>
  ALPAKA_FN_HOST constexpr T reduce(
      auto&& anyWithApi,
      ForwardIterator first,
      ForwardIterator last,
      T init,
      BinaryOperation op)
  {
    using Api = ALPAKA_TYPEOF(alpaka::getApi(anyWithApi));
      using Tag = typename AlgoDispatchTag<Api>::type;

    return AlgorithmDispatch::Op<Tag>{}.reduce(
        Api{},
        first,
        last,
        init,
        op);
  }

} // namespace clue::internal::algorithm
