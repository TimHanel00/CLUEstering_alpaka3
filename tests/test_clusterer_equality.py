'''
Test that the equality operator for clusterer objects works correctly
'''

import sys
import pandas as pd
import pytest
import numpy as np
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue
from CLUEstering import canonicalize
@pytest.fixture
def sissa():
    '''
    Returns the dataframe containing the sissa ataset
    '''
    return pd.read_csv("../data/sissa_1000.csv")




@pytest.fixture
def toy_det():
    '''
    Returns the dataframe containing the toy detector dataset
    '''
    return pd.read_csv("../data/toyDetector_1000.csv")


def test_clusterer_equality(sissa, toy_det, backend):
    '''
    Test the equality operator for clusterer objects
    '''
    # Sissa dataset
    clust1 = clue.clusterer(20., 10., 20.)
    clust1.read_data(sissa)
    clust1.run_clue(backend=backend)

    # Create a copy of the sissa clusterer to check the equality of clusterers
    clust1_copy = clue.clusterer(20., 10., 20.)
    clust1_copy.read_data(sissa)
    clust1_copy.run_clue(backend=backend)

    # toyDet dataset
    clust2 = clue.clusterer(5., 2.5, 5.)
    clust2.read_data(toy_det)
    clust2.run_clue(backend=backend)

    # Create a copy to check the equality of clusterers
    clust2_copy = clue.clusterer(5., 2.5, 5.)
    clust2_copy.read_data(toy_det)
    clust2_copy.run_clue(backend=backend)
    # Check equality
    assert np.array_equal(canonicalize(clust1.cluster_ids),
                          canonicalize(clust1_copy.cluster_ids))
    assert np.array_equal(canonicalize(clust2.cluster_ids),
                          canonicalize(clust2_copy.cluster_ids))
    # Check inequality
    assert not np.array_equal(canonicalize(clust1.cluster_ids),
                          canonicalize(clust2_copy.cluster_ids))