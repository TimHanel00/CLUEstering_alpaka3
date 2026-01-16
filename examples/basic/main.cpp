
#include <CLUEstering/CLUEstering.hpp>

#include <iostream>

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path_to_csv>\n";
    return 1;
  }

  const char* csv_path = argv[1];

  // Obtain the queue, which is used for allocations and kernel launches.
  auto queue = clue::get_queue(0U,alpaka::queueKind::blocking);
  auto device = queue.getDevice();

  // Allocate the points on the host and device.
  clue::PointsHost<2> h_points =
      clue::read_csv<2>(queue, csv_path);

  auto d_points =
      clue::PointsDevice<2, ALPAKA_TYPEOF(device)>(device, h_points.size());

  // Define the parameters for the clustering
  const float dc = 20.f, rhoc = 10.f, outlier = 20.f;

  clue::Clusterer<ALPAKA_TYPEOF(queue), 2> algo(queue, dc, rhoc, outlier);

  // Launch the clustering
  algo.make_clusters(queue, h_points, d_points);
  alpaka::onHost::wait(queue);

  // Read the results
  auto clusters_indexes = h_points.clusterIndexes();
}

