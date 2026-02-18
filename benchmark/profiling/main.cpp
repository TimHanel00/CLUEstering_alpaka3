

#include <alpaka/alpaka.hpp>
#include <algorithm>
#include <chrono>
#include <vector>

#include "CLUEstering/CLUEstering.hpp"

void run(const std::string& input_file) {
  const auto device = clue::get_device(0u);
  clue::Queue queue(device);
  auto device=queue.getDevice();
  auto dim=clue::Dim<2>{};
  auto h_points = clue::read_csv(dim,input_file);

  clue::PointsDevice d_points(device,dim, h_points.size());

  const float dc{1.5f}, rhoc{10.f}, outlier{1.5f};

  clue::Clusterer algo(queue, dim, dc, rhoc, outlier);

  algo.make_clusters(queue, h_points, d_points);
  auto clusters = algo.getClusters(h_points);
}

int main(int, char* argv[]) {
  auto input_file{std::string(argv[1])};
  run(input_file);
}
