/// @file PointsDevice.hpp
/// @brief Provides the PointsDevice class for managing points on a device
/// @authors Simone Balducci, Felice Pantaleo, Marco Rovere, Wahid Redjeb, Aurora Perego, Francesco Giacomini

#pragma once

#include "CLUEstering/data_structures/internal/PointsCommon.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"
#include "CLUEstering/detail/Dim.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <alpaka/alpaka.hpp>

namespace clue {

  template <concepts::Queue TQueue, std::size_t Ndim>
  inline void copyToHost(TQueue& queue,
                         PointsHost<Ndim>& h_points,
                         const PointsDevice<DevType<TQueue>,Ndim>& d_points) {
    if (h_points.m_size != d_points.m_size) {
      throw std::invalid_argument("copyToHost: size mismatch between host and device points");
    }

    auto const extent = alpaka::Vec{static_cast<std::size_t>(h_points.m_size)};

    // coords
    for (std::size_t dim = 0; dim < Ndim; ++dim) {
      auto dst = alpaka::makeView(alpaka::api::host, h_points.m_view.coords[dim], extent);
      auto src = alpaka::makeView(queue, d_points.m_view.coords[dim], extent);
      alpaka::onHost::memcpy(queue, dst, src);
    }

    // weight
    {
      auto dst = alpaka::makeView(alpaka::api::host, h_points.m_view.weight, extent);
      auto src = alpaka::makeView(queue, d_points.m_view.weight, extent);
      alpaka::onHost::memcpy(queue, dst, src);
    }

    // cluster index
    {
      auto dst = alpaka::makeView(alpaka::api::host, h_points.m_view.cluster_index, extent);
      auto src = alpaka::makeView(queue, d_points.m_view.cluster_index, extent);
      alpaka::onHost::memcpy(queue, dst, src);
    }

    // Make host-side data immediately usable
    alpaka::onHost::wait(queue);

    // propagate clustered state + invalidate cached cluster props on host
    h_points.m_clustered = d_points.m_clustered;
    h_points.m_clusterProperties.reset();
    h_points.m_nclusters.reset();
  }

  template <concepts::Queue TQueue, std::size_t Ndim>
  inline auto copyToHost(TQueue& queue, const PointsDevice<DevType<TQueue>,Ndim>& d_points) {
    PointsHost<Ndim> h_points{d_points.m_size};
    copyToHost(queue, h_points, d_points);
    return h_points;
  }
  template <concepts::Queue TQueue, std::size_t Ndim>
  inline void copyToDevice(TQueue& queue,
                           PointsDevice<DevType<TQueue>,Ndim>& d_points,
                           const PointsHost<Ndim>& h_points) {
    if (h_points.m_size != d_points.m_size) {
      throw std::invalid_argument("copyToDevice: size mismatch between host and device points");
    }

    auto const extent = alpaka::Vec{static_cast<std::size_t>(h_points.m_size)};

    // coords
    for (std::size_t dim = 0; dim < Ndim; ++dim) {
      auto dst = alpaka::makeView(queue, d_points.m_view.coords[dim], extent);
      auto src = alpaka::makeView(alpaka::api::host, h_points.m_view.coords[dim], extent);
      alpaka::onHost::memcpy(queue, dst, src);
    }

    // weight
    {
      auto dst = alpaka::makeView(queue, d_points.m_view.weight, extent);
      auto src = alpaka::makeView(alpaka::api::host, h_points.m_view.weight, extent);
      alpaka::onHost::memcpy(queue, dst, src);
    }

    // cluster index
    {
      auto dst = alpaka::makeView(queue, d_points.m_view.cluster_index, extent);
      auto src = alpaka::makeView(alpaka::api::host, h_points.m_view.cluster_index, extent);
      alpaka::onHost::memcpy(queue, dst, src);
    }

    // preserve/propagate state + invalidate cached nclusters on device
    d_points.m_clustered = h_points.m_clustered;
    d_points.m_nclusters.reset();
  }

  template <concepts::Queue TQueue, std::size_t Ndim>
  inline auto copyToDevice(TQueue& queue, const PointsHost<Ndim>& h_points) {
    auto dev =
        queue.getDevice();  // Alpaka3 Queue::getDevice() :contentReference[oaicite:1]{index=1}
    PointsDevice<DevType<TQueue>,Ndim> d_points{dev, h_points.m_size};
    copyToDevice(queue, d_points, h_points);
    return d_points;
  }
  /// @brief The PointsDevice class is a data structure that manages points on a device.
  /// It provides methods to allocate, access, and manipulate points in device memory.
  ///
  /// @tparam Ndim The number of dimensions of the points to manage
  /// @tparam TDev The device type to use for the allocation. Defaults to clue::Device.
  template <alpaka::onHost::concepts::Device TDev,std::size_t Ndim>
  class PointsDevice : public internal::points_interface<PointsDevice<TDev,Ndim>> {
  public:
    getBufferType<TDev, std::byte> m_buffer;
    TDev m_device;
    PointsView<Ndim> m_view;
    std::optional<std::size_t> m_nclusters;
    bool m_clustered = false;
    int32_t m_size;
    /// @brief Construct a PointsDevice object
    ///
    /// @param device The device where points are allocated
    /// @param dim The number of dimensions of the points to manage
    /// @param n_points The number of points to allocate
    ///
    PointsDevice(TDev& device,Dim<Ndim> dim, int32_t n_points);

    /// @brief Construct a PointsDevice object with a pre-allocated buffer
    ///
    /// @param device The device where points are allocated
    /// @param dim The number of dimensions of the points to manage
    /// @param n_points The number of points to allocate
    /// @param buffer The buffer to use for the points
    PointsDevice(TDev& device,Dim<Ndim> dim, int32_t n_points, std::span<std::byte> buffer);

    /// @brief Constructs a container for the points allocated on the device using interleaved data
    ///
    /// @param device The device where points are allocated
    /// @param dim The number of dimensions of the points to manage
    /// @param n_points The number of points
    /// @param input The pre-allocated buffer containing interleaved coordinates and weights
    /// @param output The pre-allocated buffer to store the cluster indexes
    /// @note The input buffer must contain the coordinates and weights in an SoA format
    PointsDevice(TDev& device, Dim<Ndim> dim, int32_t n_points, std::span<float> input, std::span<int> output);

    /// @brief Constructs a container for the points allocated on the device using separate coordinate and weight buffers
    ///
    /// @param device device where points are allocated
    /// @param dim The number of dimensions of the points to manage
    /// @param n_points The number of points
    /// @param coordinates The pre-allocated buffer containing the coordinates
    /// @param weights The pre-allocated buffer containing the weights
    /// @param output The pre-allocated buffer to store the cluster indexes
    /// @note The coordinates buffer must have a size of n_points * Ndim
    PointsDevice(TDev& device,
                 ::clue::Dim<Ndim> dim,
                 int32_t n_points,
                 std::span<float> coordinates,
                 std::span<float> weights,
                 std::span<int> output);

    /// @brief Constructs a container for the points allocated on the device using interleaved data
    ///
    /// @param device device where points are allocated
    /// @param dim The number of dimensions of the points to manage
    /// @param n_points The number of points
    /// @param input The pre-allocated buffer containing interleaved coordinates and weights
    /// @param output The pre-allocated buffer to store the cluster indexes
    /// @note The input buffer must contain the coordinates and weights in an SoA format
    PointsDevice(TDev& device, Dim<Ndim> dim, int32_t n_points, float* input, int* output);

    /// @brief Constructs a container for the points allocated on the device using separate coordinate and weight buffers
    ///
    /// @param device device where points are allocated
    /// @param dim The number of dimensions of the points to manage
    /// @param n_points The number of points
    /// @param coordinates The pre-allocated buffer containing the coordinates
    /// @param weights The pre-allocated buffer containing the weights
    /// @param output The pre-allocated buffer to store the cluster indexes
    /// @note The coordinates buffer must have a size of n_points * Ndim
    PointsDevice(TDev& device,Dim<Ndim> dim, int32_t n_points, float* coordinates, float* weights, int* output);

    /// @brief Construct a PointsDevice object with a pre-allocated buffer
    ///
    /// @param device device where points are allocated
    /// @param dim The number of dimensions of the points to manage
    /// @param n_points The number of points to allocate
    /// @param buffers The buffers to use for the points
    template <concepts::Pointer... TBuffers>
    requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1)
        PointsDevice(TDev& device,Dim<Ndim> dim, int32_t n_points, TBuffers... buffers);

    PointsDevice(const PointsDevice&) = delete;
    PointsDevice& operator=(const PointsDevice&) = delete;
    PointsDevice(PointsDevice&&) = default;
    PointsDevice& operator=(PointsDevice&&) = default;
    ~PointsDevice() = default;

#ifdef CLUE_BUILD_DOXYGEN
    /// @brief Returns the number of points
    /// @return The number of points
    ALPAKA_FN_HOST int32_t size() const;
    /// @brief Returns the coordinates of the points for a specific dimension as a const span
    /// @param dim The dimension for which to get the coordinates
    /// @return A const span of the coordinates for the specified dimension
    ALPAKA_FN_HOST auto coords(size_t dim) const;
    /// @brief Returns the coordinates of the points for a specific dimension as a span
    /// @param dim The dimension for which to get the coordinates
    /// @return A span of the coordinates for the specified dimension
    ALPAKA_FN_HOST auto coords(size_t dim);
    /// @brief Returns the weights of the points as a const span
    /// @return A const span of the weights of the points
    ALPAKA_FN_HOST auto weights() const;
    /// @brief Returns the weights of the points as a span
    /// @return A span of the weights of the points
    ALPAKA_FN_HOST auto weights();
    /// @brief Returns the cluster indexes of the points as a const span
    /// @return A const span of the cluster indexes of the points
    ALPAKA_FN_HOST auto clusterIndexes() const;
    /// @brief Returns the cluster indexes of the points as a span
    /// @return A span of the cluster indexes of the points
    ALPAKA_FN_HOST auto clusterIndexes();
    /// @brief Indicates whether the points have been clustered
    /// @return True if the points have been clustered, false otherwise
    ALPAKA_FN_HOST auto clustered() const;
    /// @brief Returns the view of the points
    /// @return A const reference to the PointsView structure containing the points data
    ALPAKA_FN_HOST const auto& view() const;
    /// @brief Returns the view of the points
    /// @return A reference to the PointsView structure containing the points data
    ALPAKA_FN_HOST auto& view();
#endif

    ALPAKA_FN_HOST auto rho() const;
    ALPAKA_FN_HOST auto rho();

    ALPAKA_FN_HOST auto delta() const;
    ALPAKA_FN_HOST auto delta();

    ALPAKA_FN_HOST auto nearestHigher() const;
    ALPAKA_FN_HOST auto nearestHigher();

    ALPAKA_FN_HOST auto isSeed() const;
    ALPAKA_FN_HOST auto isSeed();

    /// @brief Teturns the cluster properties of the points
    ///
    /// @return The number of clusters reconstructed
    /// @note This value is lazily evaluated and cached upon the first call
    ALPAKA_FN_HOST const auto& n_clusters();

  private:
    inline static constexpr std::size_t Ndim_ = Ndim;

    void mark_clustered() { m_clustered = true; }

    template <concepts::Queue _TQueue, std::size_t _Ndim>
    friend class Clusterer;
    template <concepts::Queue _TQueue, std::size_t _Ndim>
    friend void copyToHost(_TQueue& queue,
                           PointsHost<_Ndim>& h_points,
                           const PointsDevice<DevType<_TQueue>,_Ndim>& d_points);
    template <concepts::Queue _TQueue, std::size_t _Ndim>
    friend void copyToDevice(_TQueue& queue,
                             PointsDevice<DevType<_TQueue>,_Ndim>& d_points,
                             const PointsHost<_Ndim>& h_points);
    friend struct internal::points_interface<PointsDevice<TDev,Ndim>>;
  };

}  // namespace clue

#include "CLUEstering/data_structures/detail/PointsDevice.hpp"
