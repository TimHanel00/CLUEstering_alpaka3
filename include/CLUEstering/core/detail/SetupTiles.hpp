
#pragma once

#include "CLUEstering/core/detail/ComputeTiles.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/internal/Tiles.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace clue::detail {

  template <typename TQueue, std::size_t Ndim, typename TDev>
  void setup_tiles(TQueue& queue,
                   std::optional<internal::Tiles<Ndim, TDev>>& tiles,
                   const PointsHost<Ndim>& points,
                   int points_per_tile,
                   const std::array<uint8_t, Ndim>& wrapped_coordinates) {
    // TODO: reconsider the way that we compute the number of tiles
    auto ntiles =
        static_cast<int32_t>(std::ceil(points.size() / static_cast<float>(points_per_tile)));
    const auto n_per_dim = static_cast<int32_t>(std::ceil(std::pow(ntiles, 1. / Ndim)));
    ntiles = static_cast<int32_t>(std::pow(n_per_dim, Ndim));

    if (!tiles.has_value()) {
      tiles = std::make_optional<internal::Tiles<Ndim, TDev>>(queue, points.size(), ntiles);
    }
    // check if tiles are large enough for current data
    if ((tiles->extents().values < static_cast<std::size_t>(points.size())) or
        (tiles->extents().keys < static_cast<std::size_t>(ntiles))) {
      tiles->initialize(queue, points.size(), ntiles, n_per_dim);
    } else {
      tiles->reset(points.size(), ntiles, n_per_dim);
    }

    auto min_max = make_host_buffer<internal::CoordinateExtremes<Ndim>>();
    auto tile_sizes = make_host_buffer<float>(Ndim);
    detail::compute_tile_size(min_max.data(), tile_sizes, points, n_per_dim);
    alpaka::onHost::memcpy(queue, tiles->m_minmax, min_max);
    alpaka::onHost::memcpy(queue, tiles->m_tilesizes, tile_sizes);
    auto view = alpaka::makeView(wrapped_coordinates);

    auto& dst = tiles->m_wrapped;
    alpaka::onHost::memcpy(queue, dst, view, alpaka::Vec<uint8_t, 1>{Ndim});
    alpaka::onHost::wait(queue);
  }

  template <concepts::Queue TQueue,
            std::size_t Ndim,
            alpaka::onHost::concepts::Device TDev = decltype(std::declval<TQueue>().getDevice())>
  void setup_tiles(TQueue& queue,
                   std::optional<internal::Tiles<Ndim, TDev>>& tiles,
                   const PointsDevice<TDev,Ndim>& points,
                   int points_per_tile,
                   const std::array<uint8_t, Ndim>& wrapped_coordinates) {
    auto ntiles =
        static_cast<int32_t>(std::ceil(points.size() / static_cast<float>(points_per_tile)));
    const auto n_per_dim = static_cast<int32_t>(std::ceil(std::pow(ntiles, 1. / Ndim)));
    ntiles = static_cast<int32_t>(std::pow(n_per_dim, Ndim));

    if (!tiles.has_value()) {
      tiles = std::make_optional<internal::Tiles<Ndim, TDev>>(queue, points.size(), ntiles);
    }
    // check if tiles are large enough for current data
    if ((tiles->extents().values < static_cast<std::size_t>(points.size())) or
        (tiles->extents().keys < static_cast<std::size_t>(ntiles))) {
      tiles->initialize(queue, points.size(), ntiles, n_per_dim);
    } else {
      tiles->reset(points.size(), ntiles, n_per_dim);
    }

    auto min_max = make_host_buffer<internal::CoordinateExtremes<Ndim>>();
    auto tile_sizes = make_host_buffer<float>(Ndim);
    detail::compute_tile_size(queue,min_max.data(), tile_sizes.getMdSpan(), points, n_per_dim);

    alpaka::onHost::memcpy(queue, tiles->m_minmax, min_max);
    alpaka::onHost::memcpy(queue, tiles->m_minmax, tile_sizes);
    alpaka::onHost::memcpy(queue, tiles->m_tilesizes, wrapped_coordinates);
  }

}  // namespace clue::detail
