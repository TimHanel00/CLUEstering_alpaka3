
#pragma once

#include "CLUEstering/data_structures/AssociationMap.hpp"
#include "CLUEstering/detail/concepts.hpp"
#include <CLUEstering/data_structures/PointsDevice.hpp>

namespace clue {

  class AssociationMapView;
  template <typename TDev>
  class AssociationMap;

  template <typename TDev>
  class Followers {
  public:
    Followers(int32_t npoints, const TDev& dev) : m_assoc(npoints, npoints, dev) {}
    template <concepts::Queue TQueue>
    Followers(int32_t npoints, TQueue& queue) : m_assoc(npoints, npoints, queue) {}

    template <concepts::Queue TQueue>
    ALPAKA_FN_HOST void initialize(int32_t npoints, TQueue& queue) {
      m_assoc.initialize(queue, npoints, npoints);
    }
    ALPAKA_FN_HOST void reset(int32_t npoints) { m_assoc.reset(npoints, npoints); }

    template <concepts::Queue TQueue, std::size_t Ndim>
    ALPAKA_FN_HOST void fill(TQueue& queue, const PointsDevice<Ndim, TDev>& d_points) {
      m_assoc.fill(queue, d_points.size(), d_points.nearestHigher());
    }

    ALPAKA_FN_HOST inline constexpr int32_t extents() const { return m_assoc.extents().values; }

    ALPAKA_FN_HOST const AssociationMapView& view() const { return m_assoc.view(); }
    ALPAKA_FN_HOST AssociationMapView& view() { return m_assoc.view(); }

  private:
    DevAssociationMap<TDev> m_assoc;
  };

  using FollowersView = ::clue::AssociationMapView;

}  // namespace clue
