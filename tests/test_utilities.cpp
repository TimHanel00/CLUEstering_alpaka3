
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
    auto dim=::clue::Dim<2>{};
    // test both signed and unsigned integer types
    const auto dev_id2 = 0u;
    auto device2=clue::DevicePool::deviceAt(dev_id2);
    auto queue2 = device2.makeQueue();
    //static_assert(std::is_same_v<decltype(queue2), clue::Queue>, "Expected type clue::Queue");
    CHECK(queue2.getDevice() == clue::DevicePool::deviceAt(0u));
    CHECK(queue2.getDevice() == queue1.getDevice());

    // check if data allocation works
    clue::PointsHost points1(dim,1000);
    auto device3=queue1.getDevice();

    auto d_points1 = clue::PointsDevice(device3, dim, points1.size());
    clue::PointsHost points2(dim,1000);

    auto device4=queue1.getDevice();
    auto d_points2 = clue::PointsDevice(device4, dim, points2.size());
    CHECK(1);
  }

  SUBCASE("Create queue using device object") {
    auto device = clue::DevicePool::deviceAt(0u);

    auto queue2 = clue::get_queue(device);
    CHECK(queue2.getDevice() == clue::DevicePool::deviceAt(0u));
    auto dim=clue::Dim<2>{};
    // check if data allocation works
    clue::PointsHost points1(dim,1000);

    auto d_points1 = clue::PointsDevice(device,dim, points1.size());
    CHECK(1);
  }
}

TEST_CASE("Test get_clusters host function") {
  auto device = clue::DevicePool::deviceAt(0u);
  auto queue=device.makeQueue();
  const auto test_file_path = std::string(TEST_DATA_DIR) + "/data_32768.csv";
  auto dim = ::clue::Dim<2>{};
  clue::PointsHost h_points = clue::read_csv(dim,test_file_path);
  const auto n_points = h_points.size();
  

  clue::PointsDevice d_points{device, dim,  n_points};

  const float dc{1.5f}, rhoc{10.f}, outlier{1.5f};
  clue::Clusterer algo(queue,dim, dc, rhoc, outlier);
  algo.make_clusters(queue, h_points, d_points);
  auto clusters = get_clusters(h_points);
}

TEST_CASE("Test get_clusters device function") {
  auto device = clue::DevicePool::deviceAt(0u);
  auto queue=device.makeQueue();
  const auto test_file_path = std::string(TEST_DATA_DIR) + "/data_32768.csv";
  auto dim = ::clue::Dim<2>{};
  clue::PointsHost<2> h_points = clue::read_csv(dim,test_file_path);
  const auto n_points = h_points.size();


  clue::PointsDevice d_points{device, dim,  n_points};

  const float dc{1.5f}, rhoc{10.f}, outlier{1.5f};
  clue::Clusterer algo(queue,dim, dc, rhoc, outlier);
  algo.make_clusters(queue, h_points, d_points);
  auto clusters = clue::get_clusters(queue, d_points);
}
