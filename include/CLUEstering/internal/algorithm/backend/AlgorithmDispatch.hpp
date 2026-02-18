

#pragma once


namespace clue::internal::algorithm{
  namespace detail {
    template <class...>
    inline constexpr bool dependent_false_v = false;
  }
  // Default Dispatch
  struct AlgorithmDispatch
  {
    template <typename T_Api>
    struct Op
    {
      // Default: fail hard if someone uses an unsupported API or you forgot a specialization.
      template <typename... Args>
      auto count_if(Args&&...) const
      {
        static_assert(detail::dependent_false_v<T_Api>,
          "AlgorithmDispatch::Op<Api>::count_if not implemented for this Alpaka API or API is not properly enabled.");
      }

      template <typename... Args>
      auto sort(Args&&...) const
      {
        static_assert(detail::dependent_false_v<T_Api>,
          "AlgorithmDispatch::Op<Api>::sort not implemented for this Alpaka API or API is not properly enabled.");
      }

      template <typename... Args>
      auto min_element(Args&&...) const
      {
        static_assert(detail::dependent_false_v<T_Api>,
          "AlgorithmDispatch::Op<Api>::min_element not implemented for this Alpaka API or API is not properly enabled.");
      }

      template <typename... Args>
      auto max_element(Args&&...) const
      {
        static_assert(detail::dependent_false_v<T_Api>,
          "AlgorithmDispatch::Op<Api>::max_element not implemented for this Alpaka API or API is not properly enabled.");
      }

      template <typename... Args>
      auto minmax_element(Args&&...) const
      {
        static_assert(detail::dependent_false_v<T_Api>,
          "AlgorithmDispatch::Op<Api>::minmax_element not implemented for this Alpaka API or API is not properly enabled.");
      }

      template <typename... Args>
      auto reduce(Args&&...) const
      {
        static_assert(detail::dependent_false_v<T_Api>,
          "AlgorithmDispatch::Op<Api>::reduce not implemented for this Alpaka API or API is not properly enabled.");
      }
    };
  };
}