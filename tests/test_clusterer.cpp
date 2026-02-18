
#include "CLUEstering/CLUEstering.hpp"
#include "CLUEstering/utils/validation.hpp"

#include <cmath>
#include <ranges>
#include <span>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test make_cluster interfaces") {
  auto device = clue::DevicePool::deviceAt(0U);
  auto queue=clue::get_queue(device);

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/data_32768.csv";
  clue::Dim<2> dim{};
  clue::PointsHost h_points = clue::read_csv(dim, test_file_path);
  const auto n_points = h_points.size();
  clue::PointsDevice d_points{device, dim,  n_points};

  const float dc{1.3f}, rhoc{10.f}, outlier{1.3f};
  clue::Clusterer algo(queue, dim, dc, rhoc, outlier);

  SUBCASE("Run clustering without passing device points") {
    algo.make_clusters(queue, h_points);

    CHECK((clue::silhouette(h_points) >= 0.9F));
  }

  SUBCASE("Run clustering without passing the queue and device points") {
    algo.make_clusters(h_points);

    CHECK(clue::silhouette(h_points) >= 0.9f);
  }
  SUBCASE("Run clustering from device points") {
    clue::copyToDevice(queue, d_points, h_points);
    algo.make_clusters(queue, d_points);
    clue::copyToHost(queue, h_points, d_points);
    alpaka::onHost::wait(queue);

    CHECK(clue::silhouette(h_points) >= 0.9f);
  }
}

TEST_CASE("Test Clusterer constructors with invalid parameters") {
  SUBCASE("Constructor with queue") {
    auto queue = clue::get_queue(0u);
    auto dim = ::clue::Dim<2>{};
    CHECK_THROWS(clue::Clusterer(queue,dim, -1.f, 10.f));
    CHECK_THROWS(clue::Clusterer(queue,dim, 1.f, -10.f));
  }
}
