import numpy as np
import pymust


def test_txdelay_plane_and_focused(probe_name):
    param = pymust.getparam(probe_name)

    plane = pymust.txdelayPlane(param, 0.1)
    focused = pymust.txdelayFocused(param, 0, 0.03)

    for delays in (plane, focused):
        assert delays.shape[-1] == param.Nelements
        assert np.isfinite(delays).all()


def test_txdelay_circular_wave_on_linear_array():
    # txdelayCircular is only defined for linear (non-curved) arrays.
    param = pymust.getparam("L11-5v")

    circular = pymust.txdelayCircular(param, 0.1, np.pi / 3)

    assert circular.shape[-1] == param.Nelements
    assert np.isfinite(circular).all()
