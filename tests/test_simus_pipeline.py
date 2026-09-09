import numpy as np
import pymust


def test_simus_rf2iq_bmode_pipeline():
    param = pymust.getparam("L11-5v")
    param.Nelements = 8

    x = np.array([0.0, 0.0])
    y = np.array([0.0, 0.0])
    z = np.array([0.2e-2, 0.5e-2])
    rc = np.array([1.0, 1.0])
    tx_delays = np.zeros(param.Nelements).reshape((1, -1))

    rf, _ = pymust.simus(x, y, z, rc, tx_delays, param)
    assert rf.ndim == 2
    assert rf.shape[1] == param.Nelements
    assert np.isfinite(rf).all()
    assert np.any(rf != 0)

    iq = pymust.rf2iq(rf, param)
    assert iq.shape == rf.shape
    assert np.iscomplexobj(iq)
    assert np.isfinite(iq).all()

    img = pymust.bmode(iq)
    assert img.shape == rf.shape
    assert img.dtype == np.uint8
