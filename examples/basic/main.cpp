
#include <CLUEstering/CLUEstering.hpp>

#include <iostream>
template <auto Dim>
auto pointsToString(clue::PointsHost<Dim> const& points_host) -> std::vector<std::string> {
  static_assert(Dim <= 4, "Only labels x,y,z,w available.");
  constexpr std::array<char, 4> ar{'x', 'y', 'z', 'w'};

  const std::size_t length = points_host.coords(0u).size();
  std::vector<std::string> s(length, "(");

  for (std::size_t dim = 0; dim < static_cast<std::size_t>(Dim); ++dim) {
    for (std::size_t i = 0; i < length; ++i) {
      s[i] += std::string(1, ar[dim]) + " : " + std::to_string(points_host.coords(dim)[i]) + ", ";

      if (dim + 1 == static_cast<std::size_t>(Dim)) {
        s[i] = s[i].substr(0, s[i].size() - 2) + ")";
      }
    }
  }
  return s;
}
/**
* @brief Human-readable sanity check using std::cout
*
* Selects @p nr_of_points distinct uniform samples from the resulting @p clusters_indexes and
* prints each point index together with its corresponding cluster index.
*/
template <auto Dim>
void sanityCheckPrint(clue::PointsHost<Dim> const& points_host, uint32_t nr_of_points = 50U) {
  auto clusters_indexes = points_host.clusterIndexes();
  std::vector<bool> visited(clusters_indexes.size(), 0);
  alpaka::rand::engine::Philox4x32x10 phil;
  alpaka::rand::distribution::UniformReal<float> dist(0, clusters_indexes.size());
  std::vector<uint32_t> randomPoints;
  randomPoints.reserve(nr_of_points);
  for (auto k : std::views::iota(0u, nr_of_points)) {
    uint32_t pointIdx = static_cast<uint32_t>(dist(phil));
    if (!visited[pointIdx]) {
      randomPoints.emplace_back(pointIdx);
    }
    visited[pointIdx] = true;
  }
  std::ranges::sort(randomPoints);
  auto points_coordsAs_strings = pointsToString(points_host);
  for (auto pointIdx : randomPoints) {
    std::cout << " Point coords: " << points_coordsAs_strings[pointIdx]
              << " with cluster idx: " << clusters_indexes[pointIdx] << std::endl;
  }
}
int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path_to_csv>\n";
    return 1;
  }

  const char* csv_path = argv[1];

  // Obtain the queue, which is used for allocations and kernel launches.
  auto queue = clue::get_queue(0U, alpaka::queueKind::blocking);
  auto device = queue.getDevice();
  // Allocate the points on the host and device.
  clue::PointsHost<2> h_points = clue::read_csv<2>(csv_path);

  auto d_points = clue::PointsDevice<2, ALPAKA_TYPEOF(device)>(device, h_points.size());

  // Define the parameters for the clustering
  const float dc = 20.f, rhoc = 10.f, outlier = 20.f;

  clue::Clusterer<ALPAKA_TYPEOF(queue),2> algo(queue, dc, rhoc, outlier);

  // Launch the clustering
  algo.make_clusters(queue, h_points, d_points);
  alpaka::onHost::wait(queue);

  // Show the results

  sanityCheckPrint(h_points);
}
