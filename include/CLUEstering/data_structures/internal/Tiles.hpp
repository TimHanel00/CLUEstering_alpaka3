
#pragma once

#include "CLUEstering/data_structures/AssociationMap.hpp"
#include "CLUEstering/data_structures/internal/TilesView.hpp"
#include "CLUEstering/data_structures/internal/CoordinateExtremes.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/detail/make_array.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdint>
#include <alpaka/Vec.hpp>
#include <alpaka/alpaka.hpp>

namespace clue::internal {

  template <std::size_t Ndim, typename TDev>
  class Tiles {
  public:
    template <::clue::concepts::Queue TQueue>
    Tiles(TQueue& queue, int32_t n_points, int32_t n_tiles)
        : m_assoc{static_cast<std::size_t>(n_points), static_cast<std::size_t>(n_tiles), queue},
          m_minmax{make_device_buffer<CoordinateExtremes<Ndim>>(queue.getDevice(),
                                                                alpaka::Vec<std::size_t, 1U>{1})},
          m_tilesizes{make_device_buffer<float>(queue.getDevice(), Ndim)},
          m_wrapped{make_device_buffer<uint8_t>(queue.getDevice(), Ndim)},
          m_ntiles{n_tiles},
          m_nperdim{static_cast<int32_t>(std::pow(n_tiles, 1.f / Ndim))},
          m_view{} {
      m_view.indexes = m_assoc.m_indexes.data();
      m_view.offsets = m_assoc.m_offsets.data();
      m_view.minmax = m_minmax.data();
      m_view.tilesizes = m_tilesizes.data();
      m_view.wrapping = m_wrapped.data();
      m_view.npoints = n_points;
      m_view.ntiles = m_ntiles;
      m_view.nperdim = m_nperdim;
    }

    const TilesView<Ndim>& view() const { return m_view; }
    TilesView<Ndim>& view() { return m_view; }

    template <::clue::concepts::Queue TQueue>
    ALPAKA_FN_HOST void initialize(TQueue& queue, int32_t npoints, int32_t ntiles, int32_t nperdim) {
      m_assoc.initialize(queue, npoints, ntiles);
      m_ntiles = ntiles;
      m_nperdim = nperdim;

      m_view.indexes = m_assoc.m_indexes.data();
      m_view.offsets = m_assoc.m_offsets.data();
      m_view.minmax = m_minmax.data();
      m_view.tilesizes = m_tilesizes.data();
      m_view.wrapping = m_wrapped.data();
      m_view.npoints = npoints;
      m_view.ntiles = ntiles;
      m_view.nperdim = nperdim;
    }

    ALPAKA_FN_HOST void reset(int32_t npoints, int32_t ntiles, int32_t nperdim) {
      m_assoc.reset(npoints, ntiles);

      m_ntiles = ntiles;
      m_nperdim = nperdim;
      m_view.indexes = m_assoc.m_indexes.data();
      m_view.offsets = m_assoc.m_offsets.data();
      m_view.minmax = m_minmax.data();
      m_view.tilesizes = m_tilesizes.data();
      m_view.wrapping = m_wrapped.data();
      m_view.npoints = npoints;
      m_view.ntiles = ntiles;
      m_view.nperdim = nperdim;
    }

    struct GetGlobalBin {
      PointsView<Ndim> pointsView;
      TilesView<Ndim> tilesView;

      ALPAKA_FN_ACC int32_t operator()(int32_t index) const {
        float coords[Ndim];
        for (auto dim = 0u; dim < Ndim; ++dim) {
          coords[dim] = pointsView.coords[dim][index];
        }

        auto bin = tilesView.getGlobalBin(coords);
        return bin;
      }
    };

    // requires(::clue::concepts::NonHostApi<DevType<TQueue>>)
    template <::clue::concepts::Queue TQueue>
    ALPAKA_FN_HOST void fill(TQueue& queue, PointsDevice<Ndim, TDev>& d_points, size_t size) {
      auto pointsView = d_points.view();
      m_assoc.fill(queue, size, GetGlobalBin{pointsView, m_view});
    }
    // template < ::clue::concepts::Queue TQueue>
    // requires(::clue::concepts::HostApi<DevType<TQueue>>)
    // ALPAKA_FN_HOST void fill(TQueue& queue, PointsDevice<Ndim, TDev>& d_points, size_t size) {
    //   std::vector<typename decltype(m_assoc)::key_type> associations(size);
    //
    //   auto pointsView = d_points.view();
    //   GetGlobalBin binFunc{pointsView, m_view};
    //
    //   for (size_t i = 0; i < size; ++i) {
    //     associations[i] = binFunc(i);
    //   }
    //
    //   m_assoc.fill(associations);
    // }

    // ALPAKA_FN_HOST auto& minMax() {
    //   return m_minmax;
    // }
    // ALPAKA_FN_HOST auto& tileSize()  {
    //   return m_tilesizes;
    // }
    // ALPAKA_FN_HOST auto& wrapped(){
    //   return m_wrapped;
    // }

    ALPAKA_FN_HOST inline constexpr auto size() const { return m_ntiles; }

    ALPAKA_FN_HOST inline constexpr auto nPerDim() const { return m_nperdim; }

    ALPAKA_FN_HOST inline constexpr auto extents() const { return m_assoc.extents(); }
    getBufferType<TDev, CoordinateExtremes<Ndim>> m_minmax;
    getBufferType<TDev, float> m_tilesizes;
    getBufferType<TDev, uint8_t> m_wrapped;

  private:
    DevAssociationMap<TDev> m_assoc;

    int32_t m_ntiles;
    int32_t m_nperdim;
    TilesView<Ndim> m_view;
  };

}  // namespace clue::internal
