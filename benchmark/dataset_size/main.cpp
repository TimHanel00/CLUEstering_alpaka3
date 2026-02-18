
#include "CLUEstering/CLUEstering.hpp"
#include "utils/generation.hpp"
#include <benchmark/benchmark.h>

#include <cstddef>

static void BM_clustering(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    auto queue = clue::get_queue(0u);
    const auto n_points = static_cast<std::size_t>(state.range(0));
    auto dim=clue::Dim<2>{};
    clue::PointsHost h_points(dim, n_points);
    clue::PointsDevice d_points(queue,dim, n_points);
    clue::utils::generateRandomData(h_points, 20, std::make_pair(-100.f, 100.f), 1.f);
    const auto dc = 1.5f, rhoc = 10.f, outlier = 1.5f;
    state.ResumeTiming();

    clue::Clusterer algo(queue,dim, dc, rhoc, outlier);
    algo.make_clusters(queue, h_points, d_points);
  }
}

BENCHMARK(BM_clustering)->RangeMultiplier(2)->Range(1 << 10, 1 << 19);
BENCHMARK_MAIN();
