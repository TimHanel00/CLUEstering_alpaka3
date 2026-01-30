
#pragma once

#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/data_structures/internal/PointsCommon.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include "CLUEstering/internal/algorithm/reduce/reduce.hpp"
#include "CLUEstering/internal/alpaka/memory.hpp"
#include "CLUEstering/internal/meta/apply.hpp"
#include "CLUEstering/internal/alpaka/func.hpp"
#include <alpaka/alpaka.hpp>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <tuple>

namespace clue {

  namespace soa::device {

    template <std::size_t Ndim>
    inline auto computeSoASize(int32_t n_points) {
      if (n_points <= 0) {
        throw std::invalid_argument(
            "Number of points passed to PointsDevice constructor must be positive.");
      }
      return ((Ndim + 2) * sizeof(float) + 3 * sizeof(int)) * n_points;
    }
    template <concepts::Queue TQueue, std::size_t Ndim>
    void copyToHost(TQueue& queue,
                    PointsHost<Ndim>& h_points,
                    const PointsDevice<Ndim, DevType<TQueue>>& d_points);

    template <concepts::Queue TQueue, std::size_t Ndim>
    void copyToDevice(TQueue& queue,
                      PointsDevice<Ndim, DevType<TQueue>>& d_points,
                      const PointsHost<Ndim>& h_points);
    template <std::size_t Ndim>
    inline void partitionSoAView(PointsView<Ndim>& view, std::byte* buffer, int32_t n_points) {
      meta::apply<Ndim>([&]<std::size_t Dim>() {
        view.coords[Dim] = reinterpret_cast<float*>(buffer + Dim * n_points * sizeof(float));
      });
      view.weight = reinterpret_cast<float*>(buffer + Ndim * n_points * sizeof(float));
      view.cluster_index = reinterpret_cast<int*>(buffer + (Ndim + 1) * n_points * sizeof(float));
      view.is_seed = reinterpret_cast<int*>(buffer + (Ndim + 2) * n_points * sizeof(float));
      view.rho = reinterpret_cast<float*>(buffer + (Ndim + 3) * n_points * sizeof(float));
      view.nearest_higher = reinterpret_cast<int*>(buffer + (Ndim + 4) * n_points * sizeof(float));
      view.n = n_points;
    }
    template <std::size_t Ndim>
    inline void partitionSoAView(PointsView<Ndim>& view,
                                 std::byte* alloc_buffer,
                                 std::byte* buffer,
                                 int32_t n_points) {
      meta::apply<Ndim>([&]<std::size_t Dim>() {
        view.coords[Dim] = reinterpret_cast<float*>(buffer + Dim * n_points * sizeof(float));
      });
      view.weight = reinterpret_cast<float*>(buffer + Ndim * n_points * sizeof(float));
      view.cluster_index = reinterpret_cast<int*>(buffer + (Ndim + 1) * n_points * sizeof(float));
      view.is_seed = reinterpret_cast<int*>(alloc_buffer);
      view.rho = reinterpret_cast<float*>(alloc_buffer + n_points * sizeof(float));
      view.nearest_higher = reinterpret_cast<int*>(alloc_buffer + 2 * n_points * sizeof(float));
      view.n = n_points;
    }
    template <std::size_t Ndim>
    inline void partitionSoAView(PointsView<Ndim>& view,
                                 std::byte* alloc_buffer,
                                 int32_t n_points,
                                 std::span<float> coordinates,
                                 std::span<float> weights,
                                 std::span<int> output) {
      meta::apply<Ndim>(
          [&]<std::size_t Dim>() { view.coords[Dim] = coordinates.data() + Dim * n_points; });
      view.weight = weights.data();
      view.cluster_index = output.data();
      view.is_seed = reinterpret_cast<int*>(alloc_buffer);
      view.rho = reinterpret_cast<float*>(alloc_buffer + n_points * sizeof(float));
      view.nearest_higher = reinterpret_cast<int*>(alloc_buffer + 2 * n_points * sizeof(float));
      view.n = n_points;
    }
    template <std::size_t Ndim, concepts::Pointer... TBuffers>
    requires(sizeof...(TBuffers) == 3) inline void partitionSoAView(PointsView<Ndim>& view,
                                                                    std::byte* alloc_buffer,
                                                                    int32_t n_points,
                                                                    TBuffers... buffer) {
      auto buffers_tuple = std::make_tuple(buffer...);

      meta::apply<Ndim>([&]<std::size_t Dim>() {
        view.coords[Dim] = reinterpret_cast<float*>(std::get<0>(buffers_tuple) + Dim * n_points);
      });
      view.weight = std::get<1>(buffers_tuple);
      view.cluster_index = std::get<2>(buffers_tuple);
      view.is_seed = reinterpret_cast<int*>(alloc_buffer);
      view.rho = reinterpret_cast<float*>(alloc_buffer + n_points * sizeof(float));
      view.nearest_higher = reinterpret_cast<int*>(alloc_buffer + 2 * n_points * sizeof(float));
      view.n = n_points;
    }
    template <std::size_t Ndim>
    inline void partitionSoAView(PointsView<Ndim>& view,
                                 std::byte* alloc_buffer,
                                 int32_t n_points,
                                 std::span<float> input,
                                 std::span<int> output) {
      meta::apply<Ndim>([&]<std::size_t Dim>() {
        view.coords[Dim] = reinterpret_cast<float*>(input.data() + Dim * n_points);
      });
      view.weight = input.data() + Ndim * n_points;
      view.cluster_index = output.data();
      view.is_seed = reinterpret_cast<int*>(alloc_buffer);
      view.rho = reinterpret_cast<float*>(alloc_buffer + n_points * sizeof(float));
      view.nearest_higher = reinterpret_cast<int*>(alloc_buffer + 2 * n_points * sizeof(float));
      view.n = n_points;
    }
    template <std::size_t Ndim, concepts::Pointer... TBuffers>
    requires(sizeof...(TBuffers) == 2) inline void partitionSoAView(PointsView<Ndim>& view,
                                                                    std::byte* alloc_buffer,
                                                                    int32_t n_points,
                                                                    TBuffers... buffers) {
      auto buffers_tuple = std::make_tuple(buffers...);

      meta::apply<Ndim>([&]<std::size_t Dim>() {
        view.coords[Dim] = reinterpret_cast<float*>(std::get<0>(buffers_tuple) + Dim * n_points);
      });
      view.weight = std::get<0>(buffers_tuple) + Ndim * n_points;
      view.cluster_index = std::get<1>(buffers_tuple);
      view.is_seed = reinterpret_cast<int*>(alloc_buffer);
      view.rho = reinterpret_cast<float*>(alloc_buffer + n_points * sizeof(float));
      view.nearest_higher = reinterpret_cast<int*>(alloc_buffer + 2 * n_points * sizeof(float));
      view.n = n_points;
    }
    template <std::size_t Ndim, concepts::Pointer... TBuffers>
    requires(sizeof...(TBuffers) == Ndim + 2 and Ndim > 1) inline void partitionSoAView(
        PointsView<Ndim>& view, std::byte* alloc_buffer, int32_t n_points, TBuffers... buffers) {
      auto buffers_tuple = std::make_tuple(buffers...);

      meta::apply<Ndim>([&]<std::size_t Dim>() {
        view.coords[Dim] = (std::get<Dim>(buffers_tuple) + Dim * n_points);
      });
      view.weight = std::get<Ndim>(buffers_tuple) + Ndim * n_points;
      view.cluster_index = std::get<Ndim + 1>(buffers_tuple);
      view.is_seed = reinterpret_cast<int*>(alloc_buffer);
      view.rho = reinterpret_cast<float*>(alloc_buffer + n_points * sizeof(float));
      view.nearest_higher = reinterpret_cast<int*>(alloc_buffer + 2 * n_points * sizeof(float));
      view.n = n_points;
    }

  }  // namespace soa::device

  template <std::size_t Ndim, alpaka::onHost::concepts::Device TDev>
  inline PointsDevice<Ndim, TDev>::PointsDevice(TDev& device, int32_t n_points)
      : m_buffer{make_device_buffer<std::byte>(device,
                                               soa::device::computeSoASize<Ndim>(n_points))},
        m_view{},
        m_device(device),
        m_size{n_points} {
    soa::device::partitionSoAView<Ndim>(m_view, m_buffer.data(), n_points);
  }

  template <std::size_t Ndim, alpaka::onHost::concepts::Device TDev>
  inline PointsDevice<Ndim, TDev>::PointsDevice(TDev& device,
                                                int32_t n_points,
                                                std::span<std::byte> buffer)
      : m_buffer{make_device_buffer<std::byte>(device, 3 * n_points * sizeof(float))},
        m_view{},
        m_device(device),
        m_size{n_points} {
    soa::device::partitionSoAView<Ndim>(m_view, m_buffer.data(), buffer.data(), n_points);
  }

  template <std::size_t Ndim, alpaka::onHost::concepts::Device TDev>
  inline PointsDevice<Ndim, TDev>::PointsDevice(TDev& device,
                                                int32_t n_points,
                                                std::span<float> input,
                                                std::span<int> output)
      : m_buffer{make_device_buffer<std::byte>(device, 3 * n_points * sizeof(float))},
        m_view{},
        m_device(device),
        m_size{n_points} {
    soa::device::partitionSoAView<Ndim>(m_view, m_buffer.data(), n_points, input, output);
  }

  template <std::size_t Ndim, alpaka::onHost::concepts::Device TDev>
  inline PointsDevice<Ndim, TDev>::PointsDevice(TDev& device,
                                                int32_t n_points,
                                                std::span<float> coordinates,
                                                std::span<float> weights,
                                                std::span<int> output)
      : m_buffer{make_device_buffer<std::byte>(device, 3 * n_points * sizeof(float))},
        m_view{},
        m_device(device),
        m_size{n_points} {
    soa::device::partitionSoAView<Ndim>(
        m_view, m_buffer.data(), n_points, coordinates, weights, output);
  }

  template <std::size_t Ndim, alpaka::onHost::concepts::Device TDev>
  inline PointsDevice<Ndim, TDev>::PointsDevice(TDev& device,
                                                int32_t n_points,
                                                float* input,
                                                int* output)
      : m_buffer{make_device_buffer<std::byte>(device, 3 * n_points * sizeof(float))},
        m_view{},
        m_device(device),
        m_size{n_points} {
    soa::device::partitionSoAView<Ndim>(m_view, m_buffer.data(), n_points, input, output);
  }

  template <std::size_t Ndim, alpaka::onHost::concepts::Device TDev>
  inline PointsDevice<Ndim, TDev>::PointsDevice(
      TDev& device, int32_t n_points, float* coordinates, float* weights, int* output)
      : m_buffer{make_device_buffer<std::byte>(device, 3 * n_points * sizeof(float))},
        m_view{},
        m_device(device),
        m_size{n_points} {
    soa::device::partitionSoAView<Ndim>(
        m_view, m_buffer.data(), n_points, coordinates, weights, output);
  }

  template <std::size_t Ndim, alpaka::onHost::concepts::Device TDev>
  template <concepts::Pointer... TBuffers>
  requires(sizeof...(TBuffers) == Ndim + 2 and
           Ndim > 1) inline PointsDevice<Ndim, TDev>::PointsDevice(TDev& device,
                                                                   int32_t n_points,
                                                                   TBuffers... buffers)
      : m_buffer{make_device_buffer<std::byte>(device, 3 * n_points * sizeof(float))},
        m_view{},
        m_device(device),
        m_size{n_points} {
    soa::device::partitionSoAView<Ndim>(m_view, m_buffer.data(), n_points, buffers...);
  }

  template <std::size_t Ndim, alpaka::onHost::concepts::Device TDev>
  ALPAKA_FN_HOST inline auto PointsDevice<Ndim, TDev>::rho() const {
    return std::span<const float>(m_view.rho, m_size);
  }
  template <std::size_t Ndim, alpaka::onHost::concepts::Device TDev>
  ALPAKA_FN_HOST inline auto PointsDevice<Ndim, TDev>::rho() {
    return std::span<float>(m_view.rho, m_size);
  }

  template <std::size_t Ndim, alpaka::onHost::concepts::Device TDev>
  ALPAKA_FN_HOST inline auto PointsDevice<Ndim, TDev>::nearestHigher() const {
    return std::span<const int>(m_view.nearest_higher, m_size);
  }
  template <std::size_t Ndim, alpaka::onHost::concepts::Device TDev>
  ALPAKA_FN_HOST inline auto PointsDevice<Ndim, TDev>::nearestHigher() {
    return std::span<int>(m_view.nearest_higher, m_size);
  }

  template <std::size_t Ndim, alpaka::onHost::concepts::Device TDev>
  ALPAKA_FN_HOST inline auto PointsDevice<Ndim, TDev>::isSeed() const {
    return std::span<const int>(m_view.is_seed, m_size);
  }
  template <std::size_t Ndim, alpaka::onHost::concepts::Device TDev>
  ALPAKA_FN_HOST inline auto PointsDevice<Ndim, TDev>::isSeed() {
    return std::span<int>(m_view.is_seed, m_size);
  }

  template <std::size_t Ndim, alpaka::onHost::concepts::Device TDev>
  ALPAKA_FN_HOST inline const auto& PointsDevice<Ndim, TDev>::n_clusters() {
    auto queue = clue::get_queue(m_device);

    assert(m_clustered &&
           "The points have to be clustered before the cluster properties can be accessed");
    if (!m_nclusters.has_value()) {
      auto cluster_ids = this->clusterIndexes();
      auto cluster_id_view=alpaka::makeView(queue,static_cast<int*>(cluster_ids.data()),Vec1D{cluster_ids.size()});
      int32_t hostVal{0};
      auto hostView = alpaka::makeView(alpaka::api::host,&hostVal,alpaka::Vec{1U});
      auto devOutBuf = alpaka::onHost::allocLike(m_device, hostView);
      alpaka::onHost::reduce(queue,
                             DevicePool::exec(),
                             std::numeric_limits<int32_t>::lowest(),
                             devOutBuf,
                             internal::simdMax(),
                             cluster_id_view);
      alpaka::onHost::memcpy(queue, hostView, devOutBuf);
      alpaka::onHost::wait(queue);
      m_nclusters = hostVal + 1;
    }

    return m_nclusters.value();
  }

}  // namespace clue
