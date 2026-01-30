
#include "CLUEstering/CLUEstering.hpp"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
using TestApis =
    std::decay_t<decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::exec::enabledExecutors))>;
DOCTEST_TEST_CASE_TEMPLATE("Test validation scores on toy detector dataset",Api,TestApis) {
  auto queue = clue::get_queue(0u);

  const auto test_file_path = std::string(TEST_DATA_DIR) + "/toyDetector_1000.csv";
  clue::PointsHost<2> points = clue::read_csv<2>(test_file_path);
  clue::Clusterer<ALPAKA_TYPEOF(queue),2> clusterer(queue, 4.f, 2.5f, 4.f);
  clusterer.make_clusters(queue, points);

  SUBCASE("Test computation of silhouette score on all points singularly") {
    for (auto i = 0; i < points.size(); ++i) {
      if (points[i].cluster_index() < 0)  // Skip noise points
        continue;
      const auto silhouette = clue::silhouette(points, i);
      CHECK(silhouette >= -1.f);
      CHECK(silhouette <= 1.f);
    }
  }
  SUBCASE("Test computation of silhouette score on all dataset") {
    const auto silhouette = clue::silhouette(points);
    CHECK(silhouette >= -1.f);
    CHECK(silhouette <= 1.f);
  }
}
