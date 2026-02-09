
#pragma once

#include "CLUEstering/CLUEstering.hpp"
#include <vector>

template <uint8_t Ndim, typename Kernel>
void run(float dc,
         float rhoc,
         float dm,
         float seed_dc,
         int pPBin,
         std::vector<uint8_t>&& wrapped,
         std::tuple<float*, int*>&& pData,
         int32_t n_points,
         const Kernel& kernel,
         clue::concepts::Queue auto queue,
         size_t block_size) {
  auto device=queue.getDevice();
  clue::Clusterer<ALPAKA_TYPEOF(queue),Ndim> algo(queue, dc, rhoc, dm, seed_dc, pPBin);
  algo.setWrappedCoordinates(std::move(wrapped));
  // Create the host and device points
  clue::PointsHost<Ndim> h_points(n_points, std::get<0>(pData), std::get<1>(pData));
  clue::PointsDevice<Ndim,ALPAKA_TYPEOF(device)> d_points(device, n_points);

  algo.make_clusters(queue, h_points, d_points, clue::EuclideanMetric<Ndim>{}, kernel, block_size);
}
