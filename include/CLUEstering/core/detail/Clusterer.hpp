
#pragma once
#include "CLUEstering/core/DistanceMetrics.hpp"
#include "CLUEstering/core/ConvolutionalKernel.hpp"
#include "CLUEstering/core/detail/ClusteringKernels.hpp"
#include "CLUEstering/core/detail/SetupFollowers.hpp"
#include "CLUEstering/core/detail/SetupSeeds.hpp"
#include "CLUEstering/core/detail/SetupTiles.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/Followers.hpp"

#include "CLUEstering/utils/get_clusters.hpp"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace clue {
  template<concepts::Queue TQueue,std::size_t size>
  class Clusterer;
  template <concepts::Queue TQueue, std::size_t Ndim>
  Clusterer<TQueue, Ndim>::Clusterer(
      float dc, float rhoc, std::optional<float> dm, std::optional<float> seed_dc, int pPBin)
      : m_dc{dc},
        m_seed_dc{seed_dc.value_or(dc)},
        m_rhoc{rhoc},
        m_dm{dm.value_or(dc)},
        m_pointsPerTile{pPBin},
        m_wrappedCoordinates{} {
    if(m_dc <= 0.f || m_rhoc < 0.f || m_dm <= 0.f || m_seed_dc <= 0.f || m_pointsPerTile <= 0) {
      throw std::invalid_argument(
          "Invalid clustering parameters. The parameters must be positive.");
    }
  }

  template <concepts::Queue TQueue, std::size_t Ndim>
  Clusterer<TQueue, Ndim>::Clusterer(
      TQueue& /*queue*/,
      float dc,
      float rhoc,
      std::optional<float> dm,
      std::optional<float> seed_dc,
      int pPBin)
      : Clusterer(dc, rhoc, dm, seed_dc, pPBin) {}

  template <concepts::Queue TQueue,std::size_t Ndim>
  void Clusterer<TQueue,Ndim>::setParameters(
      float dc, float rhoc, std::optional<float> dm, std::optional<float> seed_dc, int pPBin) {
    m_dc = dc;
    m_dm = dm.value_or(dc);
    m_seed_dc = seed_dc.value_or(dc);
    m_rhoc = rhoc;
    m_pointsPerTile = pPBin;

    if (m_dc <= 0.f || m_rhoc < 0.f || m_dm <= 0.f || m_seed_dc <= 0.f || m_pointsPerTile <= 0) {
      throw std::invalid_argument(
          "Invalid clustering parameters. The parameters must be positive.");
    }
  }

  template <concepts::Queue TQueue,std::size_t Ndim>
  template <typename Kernel, concepts::distance_metric<Ndim> DistanceMetric>
  inline void Clusterer<TQueue,Ndim>::make_clusters(TQueue & queue,
                                             TPointsHost& h_points,
                                             const DistanceMetric& metric,
                                             const Kernel& kernel,
                                             std::size_t block_size) {
    auto d_points = PointsDevice(queue, h_points.size());

    setup(queue, h_points, d_points);
    make_clusters_impl(h_points, d_points, metric, kernel, queue, block_size);
    alpaka::onHost::wait(queue);
  }

  template <concepts::Queue TQueue, std::size_t Ndim>
  template <typename Kernel, concepts::distance_metric<Ndim> DistanceMetric>
  inline void Clusterer<TQueue, Ndim>::make_clusters(
      TQueue& queue,
      TPointsDevice& dev_points,
      const DistanceMetric& metric,
      const Kernel& kernel,
      std::size_t block_size) {
    detail::setup_tiles(queue, m_tiles, dev_points, m_pointsPerTile, m_wrappedCoordinates);
    detail::setup_followers(queue, m_followers, dev_points.size());
    make_clusters_impl(dev_points, metric, kernel, queue, block_size);
    alpaka::onHost::wait(queue);
  }
  template <concepts::Queue TQueue, std::size_t Ndim>
template <typename Kernel, concepts::distance_metric<Ndim> DistanceMetric>
inline void Clusterer<TQueue, Ndim>::make_clusters(
    TQueue& queue,
    TPointsHost& h_points,
    TPointsDevice& dev_points,
    const DistanceMetric& metric,
    const Kernel& kernel,
    std::size_t block_size)
  {
    setup(queue, h_points, dev_points);
    make_clusters_impl(h_points, dev_points, metric, kernel, queue, block_size);
    alpaka::onHost::wait(queue);
  }

  template <concepts::Queue TQueue,std::size_t Ndim>
  template <std::ranges::contiguous_range TRange>
    requires std::integral<std::ranges::range_value_t<TRange>>
  inline void Clusterer<TQueue,Ndim>::setWrappedCoordinates(const TRange& wrapped_coordinates) {
    std::ranges::copy(wrapped_coordinates, m_wrappedCoordinates.begin());
  }
  template <concepts::Queue TQueue,std::size_t Ndim>
  template <std::integral... TArgs>
  inline void Clusterer<TQueue,Ndim>::setWrappedCoordinates(TArgs... wrappedCoordinates) {
    m_wrappedCoordinates = {static_cast<uint8_t>(wrappedCoordinates)...};
  }
  template <concepts::Queue TQueue,std::size_t Ndim>
  inline auto Clusterer<TQueue,Ndim>::getClusters(const TPointsHost& h_points) {
    return get_clusters(h_points);
  }

  template <concepts::Queue TQueue,std::size_t Ndim>
  inline auto Clusterer<TQueue,Ndim>::getClusters(TQueue& queue,
                                                             const TPointsDevice& d_points) {
    return get_clusters(queue, d_points);
  }

  template <concepts::Queue TQueue,std::size_t Ndim>
  template <typename Kernel, concepts::distance_metric<Ndim> DistanceMetric>
  void Clusterer<TQueue,Ndim>::make_clusters_impl(TPointsHost& h_points,
                                           TPointsDevice& dev_points,
                                           const DistanceMetric& metric,
                                           const Kernel& kernel,
                                           TQueue& queue,
                                           std::size_t block_size) {
    const std::size_t n_points = h_points.size();
    m_tiles->template fill(queue, dev_points, n_points);

    const std::size_t grid_size = alpaka::divCeil(n_points, block_size);
    auto threadSpec = alpaka::onHost::FrameSpec{grid_size, block_size};
    std::cout<<" break 1"<<'\n';
    detail::computeLocalDensity(
        queue, threadSpec, m_tiles->view(), dev_points.view(), kernel, m_dc, metric, n_points);
    std::cout<<" break 2"<<'\n';
    auto seed_candidates = 0UL;
    detail::computeNearestHighers(queue,
                                       threadSpec,
                                       m_tiles->view(),
                                       dev_points.view(),
                                       m_dm,
                                       metric,
                                       seed_candidates,
                                       n_points);
    std::cout<<" break 3"<<'\n';
    detail::setup_seeds(queue, m_seeds, seed_candidates);
    std::cout<<" break 4"<<'\n';
    detail::findClusterSeeds(queue,
                                  threadSpec,
                                  m_seeds.value(),
                                  dev_points.view(),
                                  m_seed_dc,
                                  metric,
                                  m_rhoc,
                                  n_points);
    std::cout<<" break 5"<<'\n';

    m_followers->template fill(queue, dev_points);
    std::cout<<" break 6"<<'\n';
    detail::assignPointsToClusters(
        queue, block_size, m_seeds.value(), m_followers->view(), dev_points.view());
    std::cout<<" break 7"<<'\n';
    copyToHost(queue, h_points, dev_points);
    h_points.mark_clustered();
    dev_points.mark_clustered();
  }

  template <concepts::Queue TQueue,std::size_t Ndim>
  template <typename Kernel, concepts::distance_metric<Ndim> DistanceMetric>
  void Clusterer<TQueue,Ndim>::make_clusters_impl(TPointsDevice& dev_points,
                                           const DistanceMetric& metric,
                                           const Kernel& kernel,
                                           TQueue& queue,
                                           std::size_t block_size) {
    const std::size_t n_points = dev_points.size();
    m_tiles->template fill(queue, dev_points, n_points);

    const std::size_t grid_size = alpaka::divCeil(n_points, block_size);
    auto work_division = alpaka::onHost::FrameSpec{grid_size, block_size};

    detail::computeLocalDensity(
        queue, work_division, m_tiles->view(), dev_points.view(), kernel, m_dc, metric, n_points);
    auto seed_candidates = 0UL;

    detail::computeNearestHighers(queue,
                                       work_division,
                                       m_tiles->view(),
                                       dev_points.view(),
                                       m_dm,
                                       metric,
                                       seed_candidates,
                                       n_points);

    detail::setup_seeds(queue, m_seeds, seed_candidates);

    detail::findClusterSeeds(queue,
                                  work_division,
                                  m_seeds.value(),
                                  dev_points.view(),
                                  m_seed_dc,
                                  metric,
                                  m_rhoc,
                                  n_points);

    m_followers->template fill(queue, dev_points);

    detail::assignPointsToClusters(
        queue, block_size, m_seeds.value(), m_followers->view(), dev_points.view());

    alpaka::onHost::wait(queue);
    dev_points.mark_clustered();

  }

}  // namespace clue
