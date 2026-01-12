
#pragma once

#include "CLUEstering/internal/alpaka/config.hpp"
#include "CLUEstering/internal/alpaka/work_division.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include <alpaka/alpaka.hpp>

namespace clue {

  template <std::integral T>
  constexpr bool isPowerOf2(T v) {
    // returns true iif v has only one bit set.
    while (v) {
      if (v & 1)
        return !(v >> 1);
      else
        v >>= 1;
    }
    return false;
  }

  template <typename TAcc, typename T>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void warpPrefixScan(
      const TAcc& acc, T const* ci, T* co, int32_t i, bool active = true) {
    // ci and co may be the same
    T x = active ? ci[i] : 0;
    auto const laneId = static_cast<std::int32_t>(alpaka::onAcc::warp::getLaneIdx(acc));
    for (int32_t offset = 1; offset < acc.getExtentsOf(alpaka::onAcc::origin::warp, alpaka::onAcc::unit::threads); offset <<= 1) {
      // Force the exact type for integer types otherwise the compiler will find the template resolution ambiguous.
      using dataType = std::conditional_t<std::is_floating_point_v<T>, T, std::int32_t>;
      T y = alpaka::onAcc::warp::shflUp(acc, static_cast<dataType>(x), offset);
      if (laneId >= offset)
        x += y;
    }
    if (active)
      co[i] = x;
  }

  template <typename TAcc, typename T>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void warpPrefixScan(
      const TAcc& acc, T* c, int32_t i, bool active = true) {
    warpPrefixScan(acc, c, c, i, active);
  }

  // limited to warpSize² elements
  template <typename TAcc, typename T>
  ALPAKA_FN_ACC ALPAKA_FN_INLINE void blockPrefixScan(
      const TAcc& acc, T const* ci, T* co, int32_t size, T* ws = nullptr) {
    if constexpr (alpaka::getDeviceKind(acc)!=alpaka::deviceKind::cpu/*cpus contain only one thread*/) {
      const auto warpSize = alpaka::onAcc::warp::getSize(acc);
      auto const blockDimension(acc.getExtentsOf(alpaka::onAcc::origin::block,alpaka::onAcc::unit::threads));
      auto const blockThreadIdx(acc.getIdxWithin(alpaka::onAcc::origin::block,alpaka::onAcc::unit::threads));
      ALPAKA_ASSERT_ACC(ws);
      ALPAKA_ASSERT_ACC(size <= warpSize * warpSize);
      ALPAKA_ASSERT_ACC(0 == blockDimension % warpSize);
      auto first = blockThreadIdx;
      ALPAKA_ASSERT_ACC(isPowerOf2(warpSize));
      auto laneId = blockThreadIdx & (warpSize - 1);
      auto warpUpRoundedSize = (size + warpSize - 1) / warpSize * warpSize;

      for (auto i = first; i < warpUpRoundedSize; i += blockDimension) {
        // When padding the warp, warpPrefixScan is a noop
        warpPrefixScan(acc, laneId, ci, co, i, i < size);
        if (i < size) {
          // Skipped in warp padding threads.
          auto warpId = i / warpSize;
          ALPAKA_ASSERT_ACC(warpId < warpSize);
          if ((warpSize - 1) == laneId)
            ws[warpId] = co[i];
        }
      }
      alpaka::onAcc::syncBlockThreads(acc);
      if (size <= warpSize)
        return;
      if (blockThreadIdx < warpSize) {
        warpPrefixScan(acc, laneId, ws, blockThreadIdx);
      }
      alpaka::onAcc::syncBlockThreads(acc);
      for (auto i = first + warpSize; i < size; i += blockDimension) {
        int32_t warpId = i / warpSize;
        co[i] += ws[warpId - 1];
      }
      alpaka::onAcc::syncBlockThreads(acc);
    } else {
      co[0] = ci[0];
      for (int32_t i = 1; i < size; ++i)
        co[i] = ci[i] + co[i - 1];
    }
  }
  /**please refer to alpaka::scan for more scanning operations)
  // template <typename TAcc, typename T>
  // ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void blockPrefixScan(const TAcc& acc,
  //                                                          T* __restrict__ c,
  //                                                          int32_t size,
  //                                                          T* __restrict__ ws = nullptr) {
  //   if constexpr (alpaka::getDeviceKind(acc)==alpaka::deviceKind::cpu) {
  //     co[0] = ci[0];
  //     for(std::int32_t i = 1; i < size; ++i)
  //       co[i] = ci[i] + co[i - 1];
  //     return co
  //     const auto warpSize = alpaka::warp::getSize(acc);
  //     int32_t const blockDimension(alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u]);
  //     int32_t const blockThreadIdx(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
  //     ALPAKA_ASSERT_ACC(ws);
  //     ALPAKA_ASSERT_ACC(size <= warpSize * warpSize);
  //     ALPAKA_ASSERT_ACC(0 == blockDimension % warpSize);
  //     auto first = blockThreadIdx;
  //     auto laneId = blockThreadIdx & (warpSize - 1);
  //     auto warpUpRoundedSize = (size + warpSize - 1) / warpSize * warpSize;
  //
  //     for (auto i = first; i < warpUpRoundedSize; i += blockDimension) {
  //       // When padding the warp, warpPrefixScan is a noop
  //       warpPrefixScan(acc, laneId, c, i, i < size);
  //       if (i < size) {
  //         // Skipped in warp padding threads.
  //         auto warpId = i / warpSize;
  //         ALPAKA_ASSERT_ACC(warpId < warpSize);
  //         if ((warpSize - 1) == laneId)
  //           ws[warpId] = c[i];
  //       }
  //     }
  //     alpaka::syncBlockThreads(acc);
  //     if (size <= warpSize)
  //       return;
  //     if (blockThreadIdx < warpSize) {
  //       warpPrefixScan(acc, laneId, ws, blockThreadIdx);
  //     }
  //     alpaka::syncBlockThreads(acc);
  //     for (auto i = first + warpSize; i < size; i += blockDimension) {
  //       auto warpId = i / warpSize;
  //       c[i] += ws[warpId - 1];
  //     }
  //     alpaka::syncBlockThreads(acc);
  //   } else {
  //     for (int32_t i = 1; i < size; ++i)
  //       c[i] += c[i - 1];
  //   }
  // }
  //
  // // in principle not limited....
  // template <typename T>
  // struct multiBlockPrefixScan {
  //   template <typename TAcc>
  //   ALPAKA_FN_ACC void operator()(const TAcc& acc,
  //                                 T const* ci,
  //                                 T* co,
  //                                 std::size_t size,
  //                                 int32_t /* numBlocks */,
  //                                 int32_t* pc,
  //                                 std::size_t warpSize) const {
  //     // Get shared variable. The workspace is needed only for multi-threaded accelerators.
  //     T* ws = nullptr;
  //     if constexpr (!requires_single_thread_per_block_v<TAcc>) {
  //       ws = acc[alpaka::layer::shared];
  //     }
  //     ALPAKA_ASSERT_ACC(warpSize == static_cast<std::size_t>(alpaka::warp::getSize(acc)));
  //     [[maybe_unused]] const auto elementsPerGrid =
  //         alpaka::getWorkDiv<alpaka::Grid, alpaka::Elems>(acc)[0u];
  //     const auto elementsPerBlock = alpaka::getWorkDiv<alpaka::Block, alpaka::Elems>(acc)[0u];
  //     const auto threadsPerBlock = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
  //     const auto blocksPerGrid = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0u];
  //     const auto blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0u];
  //     const auto threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
  //     ALPAKA_ASSERT_ACC(elementsPerGrid >= size);
  //     // first each block does a scan
  //     [[maybe_unused]] int off = elementsPerBlock * blockIdx;
  //     if (size - off > 0) {
  //       blockPrefixScan(acc,
  //                       ci + off,
  //                       co + off,
  //                       std::min(elementsPerBlock, static_cast<uint32_t>(size - off)),
  //                       ws);
  //     }
  //
  //     // count blocks that finished
  //     auto& isLastBlockDone = alpaka::declareSharedVar<bool, __COUNTER__>(acc);
  //     //__shared__ bool isLastBlockDone;
  //     if (0 == threadIdx) {
  //       alpaka::mem_fence(acc, alpaka::memory_scope::Device{});
  //       auto value = alpaka::atomicAdd(acc, pc, 1, alpaka::hierarchy::Blocks{});  // block counter
  //       isLastBlockDone = (value == (int(blocksPerGrid) - 1));
  //     }
  //
  //     alpaka::syncBlockThreads(acc);
  //
  //     if (!isLastBlockDone)
  //       return;
  //
  //     ALPAKA_ASSERT_ACC(int(blocksPerGrid) == *pc);
  //
  //     // good each block has done its work and now we are left in last block
  //
  //     // let's get the partial sums from each block except the last, which receives 0.
  //     T* psum = nullptr;
  //     if constexpr (!requires_single_thread_per_block_v<TAcc>) {
  //       psum = ws + warpSize;
  //     } else {
  //       psum = alpaka::getDynSharedMem<T>(acc);
  //     }
  //     for (int32_t i = threadIdx, ni = blocksPerGrid; i < ni; i += threadsPerBlock) {
  //       auto j = elementsPerBlock * i + elementsPerBlock - 1;
  //       psum[i] = (j < size) ? co[j] : T(0);
  //     }
  //     alpaka::syncBlockThreads(acc);
  //     blockPrefixScan(acc, psum, psum, blocksPerGrid, ws);
  //
  //     // now it would have been handy to have the other blocks around...
  //     // Simplify the computation by having one version where threads per block = block size
  //     // and a second for the one thread per block accelerator.
  //     if constexpr (!requires_single_thread_per_block_v<TAcc>) {
  //       //  Here threadsPerBlock == elementsPerBlock
  //       for (std::size_t i = threadIdx + threadsPerBlock, k = 0; i < size;
  //            i += threadsPerBlock, ++k) {
  //         co[i] += psum[k];
  //       }
  //     } else {
  //       // We are single threaded here, adding partial sums starting with the 2nd block.
  //       for (std::size_t i = elementsPerBlock; i < size; i++) {
  //         co[i] += psum[i / elementsPerBlock - 1];
  //       }
  //     }
  //   }
  // };
}  // namespace clue

// declare the amount of block shared memory used by the multiBlockPrefixScan kernel
namespace alpaka::trait {

  // Variable size shared mem
  template <clue::concepts::accelerator TAcc, typename T>
  struct BlockSharedMemDynSizeBytes<clue::multiBlockPrefixScan<T>, TAcc> {
    template <typename TVec>
    ALPAKA_FN_HOST_ACC static std::size_t getBlockSharedMemDynSizeBytes(
        clue::multiBlockPrefixScan<T> const& /* kernel */,
        TVec const& /* blockThreadExtent */,
        TVec const& /* threadElemExtent */,
        T const* /* ci */,
        T const* /* co */,
        int32_t /* size */,
        int32_t numBlocks,
        int32_t const* /* pc */,
        // This trait function does not receive the accelerator object to look up the warp size
        std::size_t warpSize) {
      // We need workspace (T[warpsize]) + partial sums (T[numblocks]).
      if constexpr (clue::requires_single_thread_per_block_v<TAcc>) {
        return sizeof(T) * numBlocks;
      } else {
        return sizeof(T) * (warpSize + numBlocks);
      }
    }
  };

}  // namespace alpaka::trait
