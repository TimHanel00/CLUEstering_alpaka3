
#include "CLUEstering/CLUEstering.hpp"

#include <numeric>
#include <ranges>
#include <span>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

template <std::size_t Ndim>
struct KernelCompareDevicePoints {
  template <typename TAcc>
  ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                clue::PointsView<Ndim> view,
                                float* d_input,
                                uint32_t size,
                                int* result) const {
    for (auto [i] : alpaka::onAcc::makeIdxMap(acc,alpaka::onAcc::worker::threadsInGrid,alpaka::IdxRange{size})) {
      int comparison = 1;
      for (auto dim = 0u; dim < Ndim; ++dim) {
        comparison = (view.coords[dim][i] == d_input[i + dim * size]);
      }
      comparison = (view.weight[i] == d_input[i + Ndim * size]);

      alpaka::onAcc::atomicAnd(acc, result, comparison);
    }
  }
};

template <clue::concepts::Queue T_Queue,std::ranges::range TRange, std::size_t Ndim,typename T_Dev>
ALPAKA_FN_HOST bool compareDevicePoints(T_Queue &queue,
                                        TRange&& h_coords,
                                        TRange&& h_weights,
                                        clue::PointsDevice<Ndim,T_Dev>& d_points,
                                        uint32_t size) {
  auto h_points = clue::PointsHost<Ndim>(size);
  std::ranges::copy(h_coords, h_points.coords(0).begin());
  std::ranges::copy(h_weights, h_points.weights().begin());
  clue::copyToDevice(queue, d_points, h_points);

  // define buffers for comparison
  auto d_input = clue::make_device_buffer<float>(queue, (Ndim + 1) * size);
  alpaka::onHost::memcpy(
      queue, d_input, alpaka::makeView(alpaka::api::host,h_points.coords(0).data(), clue::Vec1D{(Ndim + 1) * size}));

  auto d_comparison_result = clue::make_device_buffer<int>(queue,clue::Vec1D{1u});
  const auto blocksize = 512u;
  const auto gridsize = alpaka::divCeil(size, blocksize);
  auto frame_spec=alpaka::onHost::FrameSpec{gridsize,blocksize};
  alpaka::onHost::fill(queue,d_comparison_result,1);
  queue.enqueue(clue::DevicePool::exec(),
                                    frame_spec,
                                    KernelCompareDevicePoints<Ndim>{},
                                    d_points.view(),
                                    d_input.data(),
                                    size,
                                    d_comparison_result.data());
  int comparison = 1;
  alpaka::onHost::memcpy(queue, alpaka::makeView(alpaka::api::host,&comparison,clue::Vec1D{1U}), d_comparison_result);
  alpaka::onHost::wait(queue);

  return static_cast<bool>(comparison);
}

TEST_CASE("Test device points with internal allocation") {
  auto device = clue::DevicePool::deviceAt(0u);
  auto queue=clue::get_queue(device);

  const uint32_t size = 1000;
  clue::PointsDevice<2,ALPAKA_TYPEOF(device)> d_points(device, size);

  auto to_float = [](int i) -> float { return static_cast<float>(i); };
  CHECK(compareDevicePoints(
      queue,
      std::views::iota(0) | std::views::take(2 * size) | std::views::transform(to_float),
      std::views::iota(0) | std::views::take(size) | std::views::transform(to_float),
      d_points,
      size));
}

TEST_CASE("Test device points with external allocation of whole buffer") {
  auto device = clue::DevicePool::deviceAt(0u);
  auto queue=clue::get_queue(device);

  const uint32_t size = 1000;
  const auto bytes = clue::soa::device::computeSoASize<2>(size);
  auto buffer = clue::make_device_buffer<std::byte>(queue, std::size_t{bytes});

  clue::PointsDevice<2,ALPAKA_TYPEOF(device)> d_points(device, size, std::span(buffer.data(), bytes));

  auto to_float = [](int i) -> float { return static_cast<float>(i); };
  CHECK(compareDevicePoints(
      queue,
      std::views::iota(0) | std::views::take(2 * size) | std::views::transform(to_float),
      std::views::iota(0) | std::views::take(size) | std::views::transform(to_float),
      d_points,
      size));
}

TEST_CASE("Test device points with external allocation passing the two buffers as spans") {
  auto device = clue::DevicePool::deviceAt(0U);
  auto queue=clue::get_queue(device);


  const uint32_t size = 1000;
  auto input = clue::make_device_buffer<float>(queue, 3 * size);
  auto output = clue::make_device_buffer<int>(queue, 2 * size);

  clue::PointsDevice<2,ALPAKA_TYPEOF(device)> d_points(
      device, size, std::span{input.data(), 3 * size}, std::span{output.data(), 2 * size});
  auto to_float = [](int i) -> float { return static_cast<float>(i); };
  CHECK(compareDevicePoints(
      queue,
      std::views::iota(0) | std::views::take(2 * size) | std::views::transform(to_float),
      std::views::iota(0) | std::views::take(size) | std::views::transform(to_float),
      d_points,
      size));
}

TEST_CASE("Test device points with external allocation passing the two buffers as pointers") {
  auto device = clue::DevicePool::deviceAt(0u);
  auto queue=clue::get_queue(device);


  const uint32_t size = 1000;
  auto input = clue::make_device_buffer<float>(queue, 3 * size);
  auto output = clue::make_device_buffer<int>(queue, 2 * size);

  clue::PointsDevice<2,ALPAKA_TYPEOF(device)> d_points(device, size, input.data(), output.data());
  auto to_float = [](int i) -> float { return static_cast<float>(i); };
  CHECK(compareDevicePoints(
      queue,
      std::views::iota(0) | std::views::take(2 * size) | std::views::transform(to_float),
      std::views::iota(0) | std::views::take(size) | std::views::transform(to_float),
      d_points,
      size));
}

TEST_CASE("Test device points with external allocation passing four buffers as spans") {
  auto device = clue::DevicePool::deviceAt(0u);
  auto queue=clue::get_queue(device);

  const uint32_t size = 1000;
  auto coords = clue::make_device_buffer<float>(queue, uint32_t{2 * size});
  auto weights = clue::make_device_buffer<float>(queue, uint32_t{size});
  auto cluster_ids = clue::make_device_buffer<int>(queue, uint32_t{size});

  clue::PointsDevice<2,ALPAKA_TYPEOF(device)> d_points(device,
                                 size,
                                 std::span{coords.data(), 2 * size},
                                 std::span{weights.data(), size},
                                 std::span{cluster_ids.data(), size});
  auto to_float = [](int i) -> float { return static_cast<float>(i); };
  CHECK(compareDevicePoints(
      queue,
      std::views::iota(0) | std::views::take(2 * size) | std::views::transform(to_float),
      std::views::iota(0) | std::views::take(size) | std::views::transform(to_float),
      d_points,
      size));
}

TEST_CASE("Test device points with external allocation passing four buffers as pointers") {
  auto device = clue::DevicePool::deviceAt(0u);
  auto queue=clue::get_queue(device);

  const uint32_t size = 1000;
  auto coords = clue::make_device_buffer<float>(queue, uint32_t{2 * size});
  auto weights = clue::make_device_buffer<float>(queue, uint32_t{size});
  auto cluster_ids = clue::make_device_buffer<int>(queue, uint32_t{size});

  clue::PointsDevice<2,ALPAKA_TYPEOF(device)> d_points(device, size, coords.data(), weights.data(), cluster_ids.data());
  auto to_float = [](int i) -> float { return static_cast<float>(i); };
  CHECK(compareDevicePoints(
      queue,
      std::views::iota(0) | std::views::take(2 * size) | std::views::transform(to_float),
      std::views::iota(0) | std::views::take(size) | std::views::transform(to_float),
      d_points,
      size));
}
///This test is errornous
// TEST_CASE("Test extrema functions on device points column") {
//
//   auto device = clue::DevicePool::deviceAt(0U);
//   auto queue=clue::get_queue(device); // gets a static queue (no lifetime issue) that will also be used internally
//
//   const uint32_t size = 1000;
//   std::vector<float> data(size);
//   std::iota(data.begin(), data.end(), 0.0f);
//   auto hostQueue=clue::DevicePool::getHost().makeQueue();
//   clue::PointsHost<2> h_points(1000);
//   alpaka::onHost::memcpy(hostQueue,h_points.coords(0),data);
//   std::ranges::copy(data, h_points.coords(0).begin());
//   std::ranges::copy(data, h_points.weights().begin());
//
//   clue::PointsDevice<2,ALPAKA_TYPEOF(device)> d_points(device, size);
//   clue::copyToDevice(queue, d_points, h_points);
//   alpaka::onHost::wait(queue);
//
//   auto max_it =
//       clue::internal::algorithm::max_element(d_points.weights().begin(), d_points.weights().end());
//   auto max = 0.f;
//   alpaka::onHost::memcpy(
//       queue, makeView(alpaka::api::host,&max,clue::Vec1D{1u}), alpaka::makeView(alpaka::getApi(queue),&(*max_it), clue::Vec1D{1u}));
//   alpaka::onHost::wait(queue);
//   CHECK(max == static_cast<float>(size - 1));
// }

TEST_CASE("Test reduction of device points column") {
  auto device = clue::DevicePool::deviceAt(0u);
  auto queue=clue::get_queue(device);

  const uint32_t size = 1000;
  std::vector<float> data(size);
  std::iota(data.begin(), data.end(), 0.0f);

  clue::PointsHost<2> h_points(1000);
  std::ranges::copy(data, h_points.coords(0).begin());
  std::ranges::copy(data, h_points.weights().begin());

  clue::PointsDevice<2,ALPAKA_TYPEOF(device)> d_points(device, size);
  clue::copyToDevice(queue, d_points, h_points);
  alpaka::onHost::wait(queue);

  CHECK(clue::internal::algorithm::reduce(d_points.weights().begin(), d_points.weights().end()) ==
        499500.0f);
}

TEST_CASE("Test constructor throwing conditions") {
  auto queue = clue::get_queue(0u);
  auto device=queue.getDevice();
  CHECK_THROWS(clue::PointsDevice<2,ALPAKA_TYPEOF(device)>(device, 0));
  CHECK_THROWS(clue::PointsDevice<2,ALPAKA_TYPEOF(device)>(device, -5));
}

TEST_CASE("Test coordinate getter throwing conditions") {
  SUBCASE("Const points") {
    const uint32_t size = 1000;
    clue::PointsHost<2> points(size);
    CHECK_THROWS(points.coords(3));
    CHECK_THROWS(points.coords(10));
  }
  SUBCASE("Non-const points") {
    const uint32_t size = 1000;
    auto queue = clue::get_queue(0u);
    auto device=queue.getDevice();
    const clue::PointsDevice<2,ALPAKA_TYPEOF(device)> points(device, size);
    CHECK_THROWS(points.coords(3));
    CHECK_THROWS(points.coords(10));
  }
}

TEST_CASE("Test n_cluster getter") {
  auto queue = clue::get_queue(0u);
  auto device=queue.getDevice();

  clue::PointsHost<2> h_points = clue::read_csv<2>(std::string(TEST_DATA_DIR) + "/data_32768.csv");
  clue::PointsDevice<2,ALPAKA_TYPEOF(device)> d_points(device, h_points.size());

  const float dc{1.3f}, rhoc{10.f}, outlier{1.3f};
  clue::Clusterer<ALPAKA_TYPEOF(queue),2> algo(queue, dc, rhoc, outlier);
  algo.make_clusters(queue, h_points, d_points);

  SUBCASE("Check the number of clusters") {
    const auto n_clusters = d_points.n_clusters();
    CHECK(n_clusters == 20);
    const auto cached_n_clusters = d_points.n_clusters();
    CHECK(cached_n_clusters == n_clusters);
  }
}
