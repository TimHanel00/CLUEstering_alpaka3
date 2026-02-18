

#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/internal/meta/apply.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include <alpaka/alpaka.hpp>

namespace clue {

  template <typename THost, typename TQueue, std::size_t Ndim>
  inline void copyToHost(THost& host_dev,
                         TQueue& queue,
                         PointsHost<Ndim>& h_points,
                         const PointsDevice<DevType<TQueue>,Ndim>& d_points) {
    alpaka::onHost::memcpy(
        queue,
        //@TODO here we can use inplace alpaka functions no wrappers (mayby think about the mapped allocation)
        alpaka::makeView(alpaka::api::host, h_points.m_view.cluster_index, Vec1D{h_points.size()}),
        //@TODO here we need alpaka functions
        alpaka::makeView(queue.getDevice(), d_points.m_view.cluster_index, Vec1D{h_points.size()}));
    h_points.mark_clustered();
  }

  template <concepts::Queue TQueue, std::size_t Ndim>
  inline auto copyToHost(TQueue& queue, const PointsDevice<DevType<TQueue>,Ndim>& d_points) {
    PointsHost<Ndim> h_points(queue, d_points.size());

    alpaka::onHost::memcpy(
        queue,
        alpaka::makeView(alpaka::api::host, h_points.m_view.cluster_index, Vec1D{h_points.size()}),
        alpaka::makeView(queue.getDevice(), d_points.m_view.cluster_index, Vec1D{h_points.size()}));
    h_points.mark_clustered();

    return h_points;
  }

  template <concepts::Queue TQueue, std::size_t Ndim>
  inline void copyToDevice(TQueue& queue,
                           PointsDevice<DevType<TQueue>,Ndim>& d_points,
                           const PointsHost<Ndim>& h_points) {
    meta::apply<Ndim>([&]<std::size_t Dim> {
      alpaka::onHost::memcpy(
          queue,
          alpaka::makeView(queue.getDevice(), d_points.m_view.coords[Dim], Vec1D{h_points.size()}),
          alpaka::makeView(
              alpaka::api::host, h_points.m_view.coords[Dim], Vec1D{Ndim * h_points.size()}));
    });
    alpaka::onHost::memcpy(
        queue,
        alpaka::makeView(queue.getDevice(), d_points.m_view.weight, Vec1D{h_points.size()}),
        alpaka::makeView(alpaka::api::host, h_points.m_view.weight, Vec1D{h_points.size()}));
  }

  template <concepts::Queue TQueue, std::size_t Ndim>
  inline auto copyToDevice(TQueue& queue, const PointsHost<Ndim>& h_points) {
    PointsDevice<DevType<TQueue>,Ndim> d_points(queue, h_points.size());

    meta::apply<Ndim>([&]<std::size_t Dim> {
      alpaka::onHost::memcpy(
          queue,
          alpaka::makeView(queue.getDevice(), d_points.m_view.coords[Dim], Vec1D{h_points.size()}),
          alpaka::makeView(
              alpaka::api::host, h_points.m_view.coords[Dim], Vec1D{Ndim * h_points.size()}));
    });
    alpaka::onHost::memcpy(
        queue,
        alpaka::makeView(queue.getDevice(), d_points.m_view.weight, Vec1D{h_points.size()}),
        alpaka::makeView(alpaka::api::host, h_points.m_view.weight, Vec1D{h_points.size()}));

    return d_points;
  }

}  // namespace clue
