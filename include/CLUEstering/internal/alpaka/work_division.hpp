
#pragma once

#include <algorithm>
#include <utility>

#include <alpaka/alpaka.hpp>

#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/detail/concepts.hpp"

using namespace alpaka_common;

namespace clue {


  /*********************************************
   *              WORKDIV CREATION
   ********************************************/

  /*
   * If the first argument is not a multiple of the second argument, round it up to the next multiple.
   */
  inline constexpr Idx round_up_by(Idx value, Idx divisor) {
    return (value + divisor - 1) / divisor * divisor;
  }

  /*
   * Return the integer division of the first argument by the second argument, rounded up to the next integer.
   */
  inline constexpr Idx divide_up_by(Idx value, Idx divisor) {
    return (value + divisor - 1) / divisor;
  }



  // /*********************************************
  //  *           RANGE COMPUTATION
  //  ********************************************/
  //
  // /*
  //  * Computes the range of the elements indexes, local to the block.
  //  * Warning: the max index is not truncated by the max number of elements of interest.
  //  */
  // template <typename TAcc>
  // ALPAKA_FN_ACC std::pair<Idx, Idx> element_index_range_in_block(const TAcc& acc,
  //                                                                const Idx elementIdxShift,
  //                                                                const unsigned int dimIndex = 0u) {
  //   // Take into account the thread index in block.
  //   const Idx threadIdxLocal(acc.getIdxWithin(alpaka::onAcc::origin::block,alpaka::onAcc::unit::threads)[dimIndex]);
  //   const Idx threadDimension(acc.getExtentsOf(alpaka::onAcc::origin::block,alpaka::onAcc::unit::threads)(acc)[dimIndex]);
  //
  //   // Compute the elements indexes in block.
  //   // Obviously relevant for CPU only.
  //   // For GPU, threadDimension == 1, and elementIdx == firstElementIdx == threadIdx + elementIdxShift.
  //   const Idx firstElementIdxLocal = threadIdxLocal * threadDimension;
  //   const Idx firstElementIdx = firstElementIdxLocal + elementIdxShift;  // Add the shift!
  //   const Idx endElementIdxUncut = firstElementIdx + threadDimension;
  //
  //   // Return element indexes, shifted by elementIdxShift.
  //   return {firstElementIdx, endElementIdxUncut};
  // }
  //
  // /*
  //  * Computes the range of the elements indexes, local to the block.
  //  * Truncated by the max number of elements of interest.
  //  */
  // template <typename TAcc>
  // ALPAKA_FN_ACC std::pair<Idx, Idx> element_index_range_in_block_truncated(
  //     const TAcc& acc,
  //     const Idx maxNumberOfElements,
  //     const Idx elementIdxShift,
  //     const unsigned int dimIndex = 0u) {
  //   // Check dimension
  //   //static_assert(alpaka::Dim<TAcc>::value == Dim1D::value,
  //   //              "Accelerator and maxNumberOfElements need to have same dimension.");
  //   auto [firstElementIdxLocal, endElementIdxLocal] =
  //       element_index_range_in_block(acc, elementIdxShift, dimIndex);
  //
  //   // Truncate
  //   endElementIdxLocal = std::min(endElementIdxLocal, maxNumberOfElements);
  //
  //   // Return element indexes, shifted by elementIdxShift, and truncated by maxNumberOfElements.
  //   return {firstElementIdxLocal, endElementIdxLocal};
  // }
  //
  // /*
  //  * Computes the range of the elements indexes in grid.
  //  * Warning: the max index is not truncated by the max number of elements of interest.
  //  */
  // template <typename TAcc>
  // ALPAKA_FN_ACC std::pair<Idx, Idx> element_index_range_in_grid(const TAcc& acc,
  //                                                               Idx elementIdxShift,
  //                                                               const unsigned int dimIndex = 0u) {
  //   // Take into account the block index in grid.
  //   const Idx blockIdxInGrid(alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[dimIndex]);
  //   const Idx blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[dimIndex]);
  //
  //   // Shift to get global indices in grid (instead of local to the block)
  //   elementIdxShift += blockIdxInGrid * blockDimension;
  //
  //   // Return element indexes, shifted by elementIdxShift.
  //   return element_index_range_in_block(acc, elementIdxShift, dimIndex);
  // }
  //
  // /*
  //  * Computes the range of the elements indexes in grid.
  //  * Truncated by the max number of elements of interest.
  //  */
  // template <concepts::accelerator TAcc>
  // ALPAKA_FN_ACC std::pair<Idx, Idx> element_index_range_in_grid_truncated(
  //     const TAcc& acc,
  //     const Idx maxNumberOfElements,
  //     Idx elementIdxShift,
  //     const unsigned int dimIndex = 0u) {
  //   // Check dimension
  //   //static_assert(dimIndex <= alpaka::Dim<TAcc>::value,
  //   //"Accelerator and maxNumberOfElements need to have same dimension.");
  //   auto [firstElementIdxGlobal, endElementIdxGlobal] =
  //       element_index_range_in_grid(acc, elementIdxShift, dimIndex);
  //
  //   // Truncate
  //   endElementIdxGlobal = std::min(endElementIdxGlobal, maxNumberOfElements);
  //
  //   // Return element indexes, shifted by elementIdxShift, and truncated by maxNumberOfElements.
  //   return {firstElementIdxGlobal, endElementIdxGlobal};
  // }
  //
  // /*
  //  * Computes the range of the element(s) index(es) in grid.
  //  * Truncated by the max number of elements of interest.
  //  */
  // template <concepts::accelerator TAcc>
  // ALPAKA_FN_ACC std::pair<Idx, Idx> element_index_range_in_grid_truncated(
  //     const TAcc& acc, const Idx maxNumberOfElements, const unsigned int dimIndex = 0u) {
  //   Idx elementIdxShift = 0u;
  //   return element_index_range_in_grid_truncated(
  //       acc, maxNumberOfElements, elementIdxShift, dimIndex);
  // }

}  // namespace clue
