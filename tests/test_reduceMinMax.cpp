  #include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/internal/MakeAssociator.hpp"

#include <ranges>
#include <span>
#include <vector>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "CLUEstering/internal/alpaka/minMax.hpp"
#include <random>
// this test does not run through smoothly on the gpu - the issue has been reported to the maintainers of alpaka
// TEST_CASE("Test reduce max (compare to stdlib)") {
//     auto device = clue::DevicePool::deviceAt(0u);
//     auto queue  = device.makeQueue();
//     std::array<int, 100> values{};
//     std::mt19937 rng{std::random_device{}()};
//     std::uniform_int_distribution<int> dist(0, 1000);
//
//     for (auto& v : values) v = dist(rng);
//     int a = 0;
//
//     auto inView  = alpaka::makeView(values);
//     auto devView= alpaka::onHost::allocLike(device, inView);
//     alpaka::onHost::memcpy(queue,devView,inView);
//     auto outView = alpaka::makeView(alpaka::api::host, &a, alpaka::Vec{1U});
//     auto devViewOut= alpaka::onHost::allocLike(device, outView);
//     alpaka::onHost::fill(queue,devViewOut,0);
//     alpaka::onHost::reduce(queue,
//                            clue::DevicePool::exec(),
//                            std::numeric_limits<int>::lowest(),
//                            devViewOut,
//                            clue::internal::simdMax(),
//                            devView);
//     alpaka::onHost::memcpy(queue,outView,devViewOut);
//     alpaka::onHost::wait(queue);
//
//     CHECK(a == *std::ranges::max_element(values));
// }
// TEST_CASE("Test reduce min") {
//   using TYPE=uint32_t;
//   auto device = clue::DevicePool::deviceAt(0u);
//   auto queue  = device.makeQueue(alpaka::queueKind::blocking);
//   std::array<TYPE, 100> values{};
//   std::mt19937 rng{std::random_device{}()};
//   std::uniform_int_distribution<TYPE> dist(0, 1000);
//
//   for (auto& v : values) v = dist(rng);
//   TYPE a = 0;
//   auto devView= alpaka::onHost::allocLike(device, values);
//   alpaka::onHost::memcpy(queue,devView,values);
//   auto outView = alpaka::makeView(alpaka::api::host, &a, alpaka::Vec{1U});
//   auto devViewOut= alpaka::onHost::allocLike(device, outView);
//   alpaka::onHost::fill(queue,devViewOut,TYPE{0});
//   alpaka::onHost::memcpy(queue,outView,devViewOut);
//   alpaka::onHost::wait(queue);
//   alpaka::onHost::reduce(queue,
//                          clue::DevicePool::exec(),
//                          std::numeric_limits<TYPE>::max(),
//                          devViewOut,
//                          clue::internal::simdMin(),
//                          devView);
//   alpaka::onHost::memcpy(queue,outView,devViewOut);
//   alpaka::onHost::wait(queue);
//   CHECK(a == *std::ranges::min_element(values));
// }
// TEST_CASE("Test reduce min (float)") {
//   using TYPE=float;
//   auto device = clue::DevicePool::deviceAt(0u);
//   auto queue  = device.makeQueue(alpaka::queueKind::blocking);
//   std::array<TYPE, 100> values{};
//   std::mt19937 rng{std::random_device{}()};
//   std::uniform_int_distribution<int> dist(0, 1000);
//
//   for (auto& v : values) v = static_cast<float>(dist(rng));
//   TYPE a = 0;
//   auto devView= alpaka::onHost::allocLike(device, values);
//   alpaka::onHost::memcpy(queue,devView,values);
//   auto outView = alpaka::makeView(alpaka::api::host, &a, alpaka::Vec{1U});
//   auto devViewOut= alpaka::onHost::allocLike(device, outView);
//   alpaka::onHost::fill(queue,devViewOut,TYPE{0});
//   alpaka::onHost::memcpy(queue,outView,devViewOut);
//   alpaka::onHost::wait(queue);
//   alpaka::onHost::reduce(queue,
//                          clue::DevicePool::exec(),
//                          std::numeric_limits<TYPE>::max(),
//                          devViewOut,
//                          clue::internal::simdMin(),
//                          devView);
//   alpaka::onHost::memcpy(queue,outView,devViewOut);
//   alpaka::onHost::wait(queue);
//   CHECK((a == *std::ranges::min_element(values)));
// }