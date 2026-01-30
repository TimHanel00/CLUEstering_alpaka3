
#pragma once
#include <alpaka/alpaka.hpp>

namespace clue {
  template <typename TDev>
  class Followers;
  namespace clue::internal {
    template <std::size_t Ndim, typename TDev>
    class Tiles;
  }
  namespace detail {
    template <typename TFunc>
    struct KernelComputeAssociations {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(
          const TAcc& acc, size_t size, int32_t* associations, int32_t nbins, TFunc func) const {
        for (auto [i] : alpaka::onAcc::makeIdxMap(
                 acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{size})) {
          int32_t bin = func(i);

          // Debug hardening (keep behind #ifdef if you want)
          if (bin < -1 || bin >= nbins) {
            printf("BAD BIN i=%d bin=%d nbins=%d\n", (int)i, bin, nbins);
            bin = -1;  // or __builtin_trap();
          }

          associations[i] = bin;
        }
      }
    };

    struct KernelComputeAssociationSizes {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    const int32_t* associations,
                                    int32_t* bin_sizes,
                                    int32_t nbins,
                                    size_t size) const {
        for (auto [i] : alpaka::onAcc::makeIdxMap(
                 acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{size})) {
          const int32_t bin_id = associations[i];
          if (bin_id >= 0 && bin_id < nbins) {
            alpaka::onAcc::atomicAdd(acc, &bin_sizes[bin_id], 1);
          }
        }
      }
    };

    struct KernelFillAssociator {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                    int32_t* indexes,
                                    const int32_t* bin_buffer,
                                    int32_t* temp_offsets,
                                    int32_t nbins,
                                    size_t size) const {
        for (auto [i] : alpaka::onAcc::makeIdxMap(
                 acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{size})) {
          const auto binId = bin_buffer[i];
          if (binId >= 0 && binId < nbins) {
            auto prev = alpaka::onAcc::atomicAdd(acc, &temp_offsets[binId], 1);
            indexes[prev] = i;
          }
        };
      }
    };  // namespace detail
  }     // namespace detail

}  // namespace clue
