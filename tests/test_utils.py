import numpy as np
import pymust


def test_genscat_generates_scatterers_within_roi():
    roi_width, roi_depth = 0.02, 0.02
    x, y, z, rc = pymust.genscat(np.array([roi_width, roi_depth]), 0.001)

    assert x.shape == y.shape == z.shape == rc.shape
    assert x.size > 0
    # The ROI is centered on x=0 with its top edge at z=0, per genscat's docstring.
    assert np.abs(x).max() <= roi_width / 2 + 1e-9
    assert (z >= -1e-9).all()
    assert z.max() <= roi_depth + 1e-9


def test_smoothn_reduces_noise(rng):
    t = np.linspace(0, 10, 200)
    clean = np.sin(t)
    noisy = clean + 0.3 * rng.standard_normal(t.size)

    smoothed, _, _ = pymust.smoothn(noisy)

    assert smoothed.shape == noisy.shape
    assert np.isfinite(smoothed).all()
    error_before = np.mean((noisy - clean) ** 2)
    error_after = np.mean((smoothed - clean) ** 2)
    assert error_after < error_before


def test_impolgrid_returns_grid_coordinates():
    param = pymust.getparam("L11-5v")

    grid = pymust.impolgrid(50, 0.05, np.pi / 3, param)

    assert len(grid) == 2
    x, z = grid
    assert x.shape == z.shape
    assert np.isfinite(x).all()
    assert np.isfinite(z).all()
