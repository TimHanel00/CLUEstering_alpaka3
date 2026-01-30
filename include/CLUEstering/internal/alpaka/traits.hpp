#pragma once
#include <alpaka/alpaka.hpp>
namespace clue::internal {
  struct MaxSimdAlpaka;
  struct MinSimdAlpaka;
}
template <>
struct alpaka::onAcc::trait::FunctorToAtomicOp<clue::internal::MaxSimdAlpaka> {
  using type = AtomicMax;
};
template <>
struct alpaka::onAcc::trait::FunctorToAtomicOp<clue::internal::MinSimdAlpaka> {
  using type = AtomicMin;
};
