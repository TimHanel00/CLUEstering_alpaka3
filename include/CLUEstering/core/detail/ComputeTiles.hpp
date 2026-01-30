
#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/CoordinateExtremes.hpp"
#include "CLUEstering/data_structures/internal/Tiles.hpp"
#include "CLUEstering/internal/algorithm/algorithm.hpp"
#include <algorithm>

namespace clue::detail {

  template <std::size_t Ndim>
  void compute_tile_size(internal::CoordinateExtremes<Ndim>* min_max,
                         alpaka::concepts::IMdSpan auto tile_sizes,
                         const PointsHost<Ndim>& h_points,
                         int32_t nPerDim) {
    for (size_t dim{}; dim != Ndim; ++dim) {
      auto coords = h_points.coords(dim);
      auto stdView = std::span<const float>(coords.data(), coords.size());
      min_max->min(dim) = *std::ranges::min_element(stdView);
      min_max->max(dim) = *std::ranges::max_element(stdView);

      const float tileSize = (min_max->max(dim) - min_max->min(dim)) / nPerDim;
      tile_sizes[dim] = tileSize;
    }
  }

  template <concepts::Queue TQueue, std::size_t Ndim, typename TDev>
  void compute_tile_size(TQueue const& queue,
                         internal::CoordinateExtremes<Ndim>* min_max,
                         alpaka::concepts::IMdSpan auto tile_sizes,
                         const PointsDevice<Ndim, TDev>& dev_points,
                         uint32_t nPerDim) {
    for (size_t dim{}; dim != Ndim; ++dim) {
      auto d_max = make_device_buffer<float>(queue, 1U);
      auto d_min = make_device_buffer<float>(queue, 1U);

      auto coords = dev_points.coords(dim);
      auto devCoordsView=alpaka::makeView(queue,coords.data(), Vec1D{coords.size()});
      alpaka::onHost::reduce(queue,
                             DevicePool::exec(),
                             std::numeric_limits<float>::lowest(),
                             d_max,
                             internal::simdMax(),
                             devCoordsView);

      alpaka::onHost::reduce(queue,
                             DevicePool::exec(),
                             std::numeric_limits<float>::max(),
                             d_min,
                             internal::simdMin(),
                             devCoordsView);
      float h_max{};
      float h_min{};
      alpaka::onHost::memcpy(queue, makeView(alpaka::api::host,&h_max,Vec1D{1U}), d_max);
      alpaka::onHost::memcpy(queue, makeView(alpaka::api::host,&h_min,Vec1D{1U}), d_min);
      min_max->min(dim) = h_min;
      min_max->max(dim) = h_max;

      tile_sizes[dim] = (h_max - h_min) / static_cast<float>(nPerDim);
    }
  }

}  // namespace clue::detail
