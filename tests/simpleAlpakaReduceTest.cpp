  #include "CLUEstering/data_structures/PointsHost.hpp"
#include "CLUEstering/data_structures/internal/MakeAssociator.hpp"

#include <ranges>
#include <span>
#include <vector>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "CLUEstering/internal/alpaka/func.hpp"

TEST_CASE("Test reduce max") {
    auto device = clue::DevicePool::deviceAt(0u);
    auto queue  = device.makeQueue();

    std::array<int, 3> values{1,2,3};
    int a = 0;

    auto inView  = alpaka::makeView(values);
    auto outView = alpaka::makeView(alpaka::api::host, &a, alpaka::Vec{1u});

    alpaka::onHost::reduce(queue,
                           clue::DevicePool::exec(),
                           std::numeric_limits<int>::lowest(),
                           outView,
                           clue::internal::simdMax(),
                           inView);

    CHECK(a == 3);
}