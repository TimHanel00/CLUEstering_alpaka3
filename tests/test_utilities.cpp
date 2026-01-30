
#include "CLUEstering/CLUEstering.hpp"

#include <cmath>
#include <ranges>
#include <span>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test clue::get_device utility") {
  clue::Device device = clue::DevicePool::deviceAt(0u);

  static_assert(std::is_same_v<decltype(device), clue::Device>, "Expected type clue::Device");
  CHECK(clue::DevicePool::devices()[0u] == device);
}

TEST_CASE("Test clue::get_queue utility") {
  SUBCASE("Create queue using device id") {
    const auto dev_id1 = 0;
    auto device=clue::DevicePool::deviceAt(dev_id1);
    auto queue1 = device.makeQueue();
    CHECK(queue1.getDevice() == clue::DevicePool::deviceAt(0u));

    // test both signed and unsigned integer types
    const auto dev_id2 = 0u;
    auto device2=clue::DevicePool::deviceAt(dev_id2);
    auto queue2 = device2.makeQueue();
    //static_assert(std::is_same_v<decltype(queue2), clue::Queue>, "Expected type clue::Queue");
    CHECK(queue2.getDevice() == clue::DevicePool::deviceAt(0u));
    CHECK(queue2.getDevice() == queue1.getDevice());

    // check if data allocation works
    clue::PointsHost<2> points1(1000);
    auto device3=queue1.getDevice();
    auto d_points1 = clue::PointsDevice<2,ALPAKA_TYPEOF(device3)>(device3, points1.size());
    clue::PointsHost<2> points2(1000);

    auto device4=queue1.getDevice();
    auto d_points2 = clue::PointsDevice<2,ALPAKA_TYPEOF(device2)>(device4, points2.size());
    CHECK(1);
  }

  SUBCASE("Create queue using device object") {
    auto device = clue::DevicePool::deviceAt(0u);

    auto queue2 = clue::get_queue(device);
    CHECK(queue2.getDevice() == clue::DevicePool::deviceAt(0u));

    // check if data allocation works
    clue::PointsHost<2> points1(1000);
    auto d_points1 = clue::PointsDevice<2,ALPAKA_TYPEOF(device)>(device, points1.size());
    CHECK(1);
  }
}

TEST_CASE("Test get_clusters host function") {
  auto device = clue::DevicePool::deviceAt(0u);
  auto queue=device.makeQueue();

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/data_32768.csv";
  clue::PointsHost<2> h_points = clue::read_csv<2>(test_file_path);
  const auto n_points = h_points.size();
  clue::PointsDevice<2,ALPAKA_TYPEOF(device)> d_points(device, n_points);

  const float dc{1.5f}, rhoc{10.f}, outlier{1.5f};
  clue::Clusterer<ALPAKA_TYPEOF(queue),2> algo(queue, dc, rhoc, outlier);
  algo.make_clusters(queue, h_points, d_points);
  auto clusters = clue::get_clusters(h_points);
}

TEST_CASE("Test get_clusters device function") {
  auto device = clue::DevicePool::deviceAt(0u);
  auto queue=device.makeQueue();

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/data_32768.csv";
  clue::PointsHost<2> h_points = clue::read_csv<2>(test_file_path);
  const auto n_points = h_points.size();
  clue::PointsDevice<2,ALPAKA_TYPEOF(device)> d_points(device, n_points);

  const float dc{1.5f}, rhoc{10.f}, outlier{1.5f};
  clue::Clusterer<ALPAKA_TYPEOF(queue),2> algo(queue, dc, rhoc, outlier);
  algo.make_clusters(queue, h_points, d_points);
  auto clusters = clue::get_clusters(queue, d_points);
}
