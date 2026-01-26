/// @file Clusterer.hpp
/// @brief Implements the Clusterer class, which is the interface for running the clustering algorithm.
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include "CLUEstering/core/DistanceMetrics.hpp"
#include "CLUEstering/core/ConvolutionalKernel.hpp"
#include "CLUEstering/core/detail/ClusteringKernels.hpp"
#include "CLUEstering/core/detail/SetupFollowers.hpp"
#include "CLUEstering/core/detail/SetupTiles.hpp"
#include "CLUEstering/data_structures/AssociationMap.hpp"
#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/Tiles.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace clue {

  /// @brief The Clusterer class is the interface for running the clustering algorithm.
  /// It provides methods to set up the clustering parameters, initializes the internal buffers
  /// and runs the clustering algorithm on host or device points.
  ///
  /// @tparam Ndim The number of dimensions of the points to cluster
  template <concepts::Queue TQueue, std::size_t Ndim>
  class Clusterer {
  private:
    using CoordinateExtremes = internal::CoordinateExtremes<Ndim>;
    using TPointsHost = PointsHost<Ndim>;
    using TDev = DevType<TQueue>;
    using TPointsDevice = PointsDevice<Ndim, TDev>;
    using TilesDevice = internal::Tiles<Ndim, TDev>;
    using FollowersDevice = Followers<TDev>;

    float m_dc;
    float m_seed_dc;
    float m_rhoc;
    float m_dm;
    int m_pointsPerTile;  // average number of points found in a tile
    std::array<uint8_t, Ndim> m_wrappedCoordinates;

    std::optional<TilesDevice> m_tiles;
    std::optional<internal::SeedArray<TDev>> m_seeds;
    std::optional<FollowersDevice> m_followers;

    void setup(TQueue& queue, const TPointsHost& h_points, TPointsDevice& dev_points) {
      detail::setup_tiles(queue, m_tiles, h_points, m_pointsPerTile, m_wrappedCoordinates);
      detail::setup_followers(queue, m_followers, h_points.size());
      copyToDevice(queue, dev_points, h_points);
      alpaka::onHost::wait(queue);
    }

    template <typename Kernel = FlatKernel,
              concepts::distance_metric<Ndim> DistanceMetric = EuclideanMetric<Ndim>>
    void make_clusters_impl(TPointsHost& h_points,
                            TPointsDevice& dev_points,
                            const DistanceMetric& metric,
                            const Kernel& kernel,
                            TQueue& queue,
                            std::size_t block_size);
    template <typename Kernel = FlatKernel,
              concepts::distance_metric<Ndim> DistanceMetric = EuclideanMetric<Ndim>>
    void make_clusters_impl(TPointsDevice& dev_points,
                            const DistanceMetric& metric,
                            const Kernel& kernel,
                            TQueue& queue,
                            std::size_t block_size);

  public:
    /// @brief Constuct a Clusterer object
    ///
    /// @param dc Distance threshold for clustering.
    /// @param rhoc Density threshold for clustering
    /// @param dm Minimum distance between clusters. This parameter is optional and by default dc is used.
    /// @param seed_dc Distance threshold for seed points. This parameter is optional and by default dc is used.
    /// @param pPBin Number of points per bin, used to determine the tile size
    Clusterer(float dc,
              float rhoc,
              std::optional<float> dm = std::nullopt,
              std::optional<float> seed_dc = std::nullopt,
              int pPBin = 128);
    /// @brief Constuct a Clusterer object
    ///
    /// @param queue The queue to use for the device operations
    /// @param dc Distance threshold for clustering.
    /// @param rhoc Density threshold for clustering
    /// @param dm Minimum distance between clusters. This parameter is optional and by default dc is used.
    /// @param seed_dc Distance threshold for seed points. This parameter is optional and by default dc is used.
    /// @param pPBin Number of points per bin, used to determine the tile size
    Clusterer(TQueue& queue,
              float dc,
              float rhoc,
              std::optional<float> dm = std::nullopt,
              std::optional<float> seed_dc = std::nullopt,
              int pPBin = 128);

    /// @brief Set the parameters for the clustering algorithm
    ///
    /// @param dc Distance threshold for clustering
    /// @param rhoc Density threshold for clustering
    /// @param dm Minimum distance between clusters. This parameter is optional and by default dc is used.
    /// @param seed_dc Distance threshold for seed points. This parameter is optional and by default dc is used.
    /// @param pPBin Number of points per bin, used to determine the tile size
    void setParameters(float dc,
                       float rhoc,
                       std::optional<float> dm = std::nullopt,
                       std::optional<float> seed_dc = std::nullopt,
                       int pPBin = 128);

    /// @brief Construct the clusters from host points
    ///
    /// @tparam Kernel The type of convolutional kernel to use
    /// @tparam DistanceMetric The type of distance metric to use
    /// @param queue The queue to use for the device operations
    /// @param h_points Host points to cluster
    /// @param metric The distance metric to use for clustering, default is EuclideanMetric
    /// @param kernel The convolutional kernel to use for computing the local densities, default is FlatKernel with height 0.5
    /// @param block_size The size of the blocks to use for clustering, default is 256
    template <typename Kernel = FlatKernel,
              concepts::distance_metric<Ndim> DistanceMetric = EuclideanMetric<Ndim>>
    void make_clusters(TQueue& queue,
                       TPointsHost& h_points,
                       const DistanceMetric& metric = EuclideanMetric<Ndim>{},
                       const Kernel& kernel = FlatKernel{.5f},
                       std::size_t block_size = 256);
    /// @brief Construct the clusters from host points
    ///
    /// @tparam Kernel The type of convolutional kernel to use
    /// @tparam DistanceMetric The type of distance metric to use
    /// @param h_points Host points to cluster
    /// @param metric The distance metric to use for clustering, default is EuclideanMetric
    /// @param kernel The convolutional kernel to use for computing the local densities, default is FlatKernel with height 0.5
    /// @param block_size The size of the blocks to use for clustering, default is 256
    /// @note This method creates a temporary queue for the operations on the device
    template <typename Kernel = FlatKernel,
              concepts::distance_metric<Ndim> DistanceMetric = EuclideanMetric<Ndim>>
    void make_clusters(TPointsHost& h_points,
                       const DistanceMetric& metric = EuclideanMetric<Ndim>{},
                       const Kernel& kernel = FlatKernel{.5f},
                       std::size_t block_size = 256);
    /// @brief Construct the clusters from host and device points
    ///
    /// @tparam Kernel The type of convolutional kernel to use
    /// @tparam DistanceMetric The type of distance metric to use
    /// @param queue The queue to use for the device operations
    /// @param h_points Host points to cluster
    /// @param dev_points Device points to cluster
    /// @param metric The distance metric to use for clustering, default is EuclideanMetric
    /// @param kernel The convolutional kernel to use for computing the local densities, default is FlatKernel with height 0.5
    /// @param block_size The size of the blocks to use for clustering, default is 256
    template <typename Kernel = FlatKernel,
              concepts::distance_metric<Ndim> DistanceMetric = EuclideanMetric<Ndim>>
    void make_clusters(TQueue& queue,
                       TPointsHost& h_points,
                       TPointsDevice& dev_points,
                       const DistanceMetric& metric = EuclideanMetric<Ndim>{},
                       const Kernel& kernel = FlatKernel{.5f},
                       std::size_t block_size = 256);
    /// @brief Construct the clusters from device points
    ///
    /// @tparam Kernel The type of convolutional kernel to use
    /// @tparam DistanceMetric The type of distance metric to use
    /// @param queue The queue to use for the device operations
    /// @param dev_points Device points to cluster
    /// @param metric The distance metric to use for clustering, default is EuclideanMetric
    /// @param kernel The convolutional kernel to use for computing the local densities, default is FlatKernel with height 0.5
    /// @param block_size The size of the blocks to use for clustering, default is 256
    template <typename Kernel = FlatKernel,
              concepts::distance_metric<Ndim> DistanceMetric = EuclideanMetric<Ndim>>
    void make_clusters(TQueue& queue,
                       TPointsDevice& dev_points,
                       const DistanceMetric& metric = EuclideanMetric<Ndim>{},
                       const Kernel& kernel = FlatKernel{.5f},
                       std::size_t block_size = 256);

    /// @brief Specify which coordinates are periodic
    ///
    /// @param wrappedCoordinates Array of wrapped coordinates, where 1 means periodic and 0 means non-periodic
    template <std::ranges::contiguous_range TRange>
    requires std::integral<std::ranges::range_value_t<TRange>>
    void setWrappedCoordinates(const TRange& wrapped_coordinates);
    /// @brief Specify which coordinates are periodic
    ///
    /// @tparam TArgs Types of the wrapped coordinates, should be convertible to uint8_t
    /// @param wrappedCoordinates Wrapped coordinates, where 1 means periodic and 0 means non-periodic
    template <std::integral... TArgs>
    void setWrappedCoordinates(TArgs... wrapped_coordinates);

    /// @brief Get the clusters from the host points
    ///
    /// @param h_points Host points
    /// @return An host associator mapping clusters and points
    auto getClusters(const TPointsHost& h_points);
    /// @brief Get the clusters from the device points
    /// This function returns an associator object mapping the clusters to the points they contain.
    ///
    /// @param d_points Device points
    /// @return An device associator mapping clusters and points
    auto getClusters(TQueue& queue, const TPointsDevice& d_points);
  };

}  // namespace clue

#include "CLUEstering/core/detail/Clusterer.hpp"
