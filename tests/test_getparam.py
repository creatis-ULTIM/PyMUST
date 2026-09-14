import numpy as np
import pymust


def test_getparam_returns_sane_probe_parameters(probe_name):
    param = pymust.getparam(probe_name)

    assert param.Nelements > 0
    assert param.fc > 0
    assert param.pitch > 0
    assert param.bandwidth > 0


def test_getparam_unknown_probe_raises():
    import pytest

    with pytest.raises(Exception):
        pymust.getparam("not-a-real-probe")
