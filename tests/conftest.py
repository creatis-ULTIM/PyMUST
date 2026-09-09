import numpy as np
import pytest


@pytest.fixture(params=["L11-5v", "L12-3v", "C5-2v", "P4-2v"])
def probe_name(request):
    return request.param


@pytest.fixture
def rng():
    return np.random.default_rng(0)
