

#pragma once

#include <alpaka/alpaka.hpp>
#include <utility>

#include "CLUEstering/internal/algorithm/backend/AlgoDispatchTag.hpp"
#include "CLUEstering/internal/algorithm/backend/backendImp.hpp"

namespace clue::internal::algorithm {

  template <typename ExecutionPolicy, typename RandomAccessIterator>
  ALPAKA_FN_HOST constexpr void sort(
      auto&& anyWithApi,
      ExecutionPolicy&& policy,
      RandomAccessIterator first,
      RandomAccessIterator last)
  {
    using Api = ALPAKA_TYPEOF(alpaka::getApi(anyWithApi));
    using Tag = typename AlgoDispatchTag<Api>::type;

    AlgorithmDispatch::Op<Tag>{}.sort(
        Api{},
        std::forward<ExecutionPolicy>(policy),
        first,
        last);
  }

  template <typename RandomAccessIterator>
  ALPAKA_FN_HOST constexpr void sort(
      auto&& anyWithApi,
      RandomAccessIterator first,
      RandomAccessIterator last)
  {
    using Api = ALPAKA_TYPEOF(alpaka::getApi(anyWithApi));
    using Tag = typename AlgoDispatchTag<Api>::type;

    AlgorithmDispatch::Op<Tag>{}.sort(
        Api{},
        first,
        last);
  }

  template <typename RandomAccessIterator, typename Compare>
  ALPAKA_FN_HOST constexpr void sort(
      auto&& anyWithApi,
      RandomAccessIterator first,
      RandomAccessIterator last,
      Compare comp)
  {
    using Api = ALPAKA_TYPEOF(alpaka::getApi(anyWithApi));
      using Tag = typename AlgoDispatchTag<Api>::type;

    AlgorithmDispatch::Op<Tag>{}.sort(
        Api{},
        first,
        last,
        comp);
  }

  template <typename ExecutionPolicy, typename RandomAccessIterator, typename Compare>
  ALPAKA_FN_HOST constexpr void sort(
      auto&& anyWithApi,
      ExecutionPolicy&& policy,
      RandomAccessIterator first,
      RandomAccessIterator last,
      Compare comp)
  {
    using Api = ALPAKA_TYPEOF(alpaka::getApi(anyWithApi));
      using Tag = typename AlgoDispatchTag<Api>::type;

    AlgorithmDispatch::Op<Tag>{}.sort(
        Api{},
        std::forward<ExecutionPolicy>(policy),
        first,
        last,
        comp);
  }

} // namespace clue::internal::algorithm