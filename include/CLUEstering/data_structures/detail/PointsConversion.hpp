

#pragma once

#include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/PointsDevice.hpp"
#include "CLUEstering/internal/meta/apply.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include <alpaka/alpaka.hpp>

namespace clue {

  template <typename THost,typename TQueue, std::size_t Ndim, alpaka::onHost::concepts::Device TDev>
  inline void copyToHost(THost &host_dev,TQueue& queue,
                         PointsHost<Ndim>& h_points,
                         const PointsDevice<Ndim, TDev>& d_points) {
    alpaka::onHost::memcpy(
        queue,
        //@TODO here we can use inplace alpaka functions no wrappers (mayby think about the mapped allocation)
        alpaka::makeView(host_dev,h_points.m_view.cluster_index, h_points.size()),
        //@TODO here we need alpaka functions
        alpaka::makeView(queue.getDevice(), d_points.m_view.cluster_index, h_points.size()));
    h_points.mark_clustered();
  }

  template <typename THost,concepts::Queue TQueue, std::size_t Ndim, alpaka::onHost::concepts::Device TDev>
  inline auto copyToHost(THost host_dev,TQueue& queue, const PointsDevice<Ndim, TDev>& d_points) {
    PointsHost<Ndim> h_points(queue, d_points.size());

    alpaka::onHost::memcpy(
        queue,
        alpaka::makeView(host_dev,h_points.m_view.cluster_index, h_points.size()),
        alpaka::makeView(queue.getDevice(), d_points.m_view.cluster_index, h_points.size()));
    h_points.mark_clustered();

    return h_points;
  }

  template <concepts::CpuDevice THost,concepts::Queue TQueue, std::size_t Ndim, alpaka::onHost::concepts::Device TDev>
  inline void copyToDevice(THost devHost,TQueue& queue,
                           PointsDevice<Ndim, TDev>& d_points,
                           const PointsHost<Ndim>& h_points) {
    meta::apply<Ndim>([&]<std::size_t Dim> {
      alpaka::onHost::memcpy(
          queue,
          make_device_view(queue.getDevice(), d_points.m_view.coords[Dim], h_points.size()),
          make_host_view(h_points.m_view.coords[Dim], Ndim * h_points.size()));
    });
    alpaka::onHost::memcpy(queue,
                   make_device_view(queue.getDevice(), d_points.m_view.weight, h_points.size()),
                   make_host_view(h_points.m_view.weight, h_points.size()));
  }

  template <concepts::CpuDevice THost,concepts::Queue TQueue, std::size_t Ndim, alpaka::onHost::concepts::Device TDev>
  inline auto copyToDevice(THost devHost,TQueue& queue, const PointsHost<Ndim>& h_points) {
    PointsDevice<Ndim, TDev> d_points(queue, h_points.size());

    meta::apply<Ndim>([&]<std::size_t Dim> {
      alpaka::onHost::memcpy(
          queue,
          make_device_view(queue.getDevice(), d_points.m_view.coords[Dim], h_points.size()),
          make_host_view(h_points.m_view.coords[Dim], Ndim * h_points.size()));
    });
    alpaka::onHost::memcpy(queue,
                   make_device_view(queue.getDevice(), d_points.m_view.weight, h_points.size()),
                   make_host_view(h_points.m_view.weight, h_points.size()));

    return d_points;
  }

}  // namespace clue
