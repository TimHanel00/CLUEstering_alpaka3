
#pragma once

#include <alpaka/alpaka.hpp>
#include "CLUEstering/internal/algorithm/backend/AlgoDispatchTag.hpp"
#include "CLUEstering/internal/algorithm/backend/backendImp.hpp"

namespace clue::internal::algorithm {

  template <typename ExecutionPolicy, typename InputIterator, typename Predicate>
  ALPAKA_FN_HOST inline constexpr auto count_if(
      auto&& anyWithApi,
      ExecutionPolicy&& policy,
      InputIterator first,
      InputIterator last,
      Predicate pred)
  {
    using Api = ALPAKA_TYPEOF(alpaka::getApi(anyWithApi));
    using Tag = typename AlgoDispatchTag<Api>::type;

    return AlgorithmDispatch::Op<Tag>{}.count_if(
        Api{},
        std::forward<ExecutionPolicy>(policy),
        first,
        last,
        pred);
  }

  template <typename InputIterator, typename Predicate>
  ALPAKA_FN_HOST inline constexpr auto count_if(
      auto&& anyWithApi,
      InputIterator first,
      InputIterator last,
      Predicate pred)
  {
    using Api = ALPAKA_TYPEOF(alpaka::getApi(anyWithApi));
    using Tag = typename AlgoDispatchTag<Api>::type;

    return AlgorithmDispatch::Op<Tag>{}.count_if(
        Api{},
        first,
        last,
        pred);
  }
}  // namespace clue::internal::algorithm
