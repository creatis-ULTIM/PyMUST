"""Fixed input scenarios shared by the golden-data generator and the regression
tests, so both sides always compute from the exact same inputs.
"""
import numpy as np
import pymust

DATA_DIR = __import__("pathlib").Path(__file__).parent / "data"


PLANE_WAVE_TILTS = [-0.3, -0.1, 0.0, 0.1, 0.3]  # radians, several steering directions


def pfield_scenario():
    param = pymust.getparam("L11-5v")
    param.Nelements = 16

    x, z = pymust.impolgrid(10, 0.04, np.pi / 4, param)
    y = np.zeros_like(x)

    outputs = {}
    for tilt in PLANE_WAVE_TILTS:
        delays = pymust.txdelayPlane(param, tilt)
        rp, _, _ = pymust.pfield(x, y, z, delays, param)
        outputs[f"rms_pressure_plane_tilt_{tilt:+.1f}"] = rp

    focused_delays = pymust.txdelayFocused(param, 0.005, 0.03)
    rp_focused, _, _ = pymust.pfield(x, y, z, focused_delays, param)
    outputs["rms_pressure_focused"] = rp_focused

    return outputs


def simus_scenario():
    param = pymust.getparam("L11-5v")
    param.Nelements = 8

    x = np.array([0.0, 0.3e-2, -0.3e-2])
    y = np.array([0.0, 0.0, 0.0])
    z = np.array([0.2e-2, 0.4e-2, 0.6e-2])
    rc = np.array([1.0, 0.7, 1.3])
    delays = pymust.txdelayPlane(param, 0.05)

    rf, _ = pymust.simus(x, y, z, rc, delays, param)
    iq = pymust.rf2iq(rf, param)
    img = pymust.bmode(iq)
    return {"rf": rf, "iq": iq, "bmode": img}


SCENARIOS = {
    "pfield": pfield_scenario,
    "simus": simus_scenario,
}
