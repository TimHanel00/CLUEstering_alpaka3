#pragma once
#include <alpaka/alpaka.hpp>
#include "CLUEstering/internal/alpaka/traits.hpp"
namespace clue::internal{
  struct MaxSimdAlpaka{

    auto operator()(auto&& x, auto&& y) const
    {
      return alpaka::math::max(ALPAKA_FORWARD(x), ALPAKA_FORWARD(y));
    }
    };
  struct MinSimdAlpaka{
    auto operator()(auto&& x, auto&& y) const {

      return alpaka::math::min(ALPAKA_FORWARD(x), ALPAKA_FORWARD(y));
    }
  };
  auto simdMin(){
    return alpaka::ScalarFunc{MinSimdAlpaka{}};
  }
  auto simdMax(){
    return alpaka::ScalarFunc{MaxSimdAlpaka{}};
  }
}