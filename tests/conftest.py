import pytest
import sys
sys.path.insert(1, '../CLUEstering/')
from CLUEstering import all_backends

def pytest_generate_tests(metafunc):
    if "backend" in metafunc.fixturenames:
        backends = all_backends()
        print(f' devices: {backends}')
        if not backends:
            pytest.skip("No devices available")
        metafunc.parametrize("backend", backends, ids=str)
