

import sys
from sklearn.metrics import silhouette_score
import argparse
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue

def main():
    args = clue.clue_args()

    c = clue.clusterer(args=args)
    c.read_data(args.input)
    if args.backend == "all":
        for b in getattr(clue, "backends", ["cpu serial"]):
            print(f"\n=== Running backend: {b} ===")
            c.run_clue(backend=b, block_size=args.block_size, device_id=args.device_id, verbose=args.verbose, dimensions=args.dimension)
    else:
        print(f'Selected backend: {args.backend}')
        c.run_clue_from_args(args)
    c.to_csv('./', 'out.csv')

    if(silhouette_score(c.coords.T, c.cluster_ids) > 0.8):
        print("Executed with no errors")
    else:
        print("The silhouette_score is very low - this could be due to parameter selection!")
    c.cluster_plotter()

if __name__ == "__main__":
    main()