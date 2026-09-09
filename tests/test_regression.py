"""Golden-master regression tests.

These compare simus/pfield output, computed fresh on every test run, against
reference arrays saved in tests/data/. simus and pfield are deterministic
within one environment (no randomness - see tests/golden_scenarios.py), but
NOT bit-reproducible across environments: pfield accumulates results in
single precision (complex64), and different BLAS/LAPACK backends (e.g.
OpenBLAS vs Apple's Accelerate) round matrix reductions differently. simus
then runs a sharp threshold ("zero out samples below -100dB relative to the
peak", see the RelThresh/tanh step in simus.py) on top of that, which turns
tiny backend-dependent noise into visibly different values for the handful
of samples that sit right at the threshold. This was caught in practice by
comparing a conda-env-generated reference against a plain pip install: exact
comparison failed on ~36% of RF samples even though every value was
numerically negligible (all well under 1e-4 of the peak amplitude).

So a sample's relative tolerance depends on how loud it is compared to the
signal's own peak (in dB, 20*log10(|value|/peak)):
  - at or above NOISE_FLOOR_DB: tight rtol - this is "real" signal, and a
    regression here (wrong scaling, wrong timing/shape, wrong physics)
    should still fail loudly.
  - below NOISE_FLOOR_DB: loose rtol - down in the noise floor, relative
    comparison is meaningless (dividing by a near-zero reference blows up
    the ratio) and this is exactly where backend rounding causes samples to
    snap across simus's threshold. A small absolute tolerance (scaled to
    the peak) still catches a sample that shouldn't be near-zero at all.

If you intentionally change the numerics of simus/pfield (e.g. a bug fix that
changes the output), regenerate the references and review the diff:

    python tests/generate_golden_data.py
"""
import numpy as np
import pytest

from golden_scenarios import DATA_DIR, SCENARIOS

NOISE_FLOOR_DB = -30.0
TIGHT_RTOL = 1e-4
LOOSE_RTOL = 1.0
ATOL_FRACTION_OF_PEAK = 1e-5


def _load_reference(name):
    path = DATA_DIR / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"no golden reference at {path}; run generate_golden_data.py")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _assert_matches_reference(fresh_value, ref_value, label):
    fresh_value = np.asarray(fresh_value, dtype=np.complex128 if np.iscomplexobj(fresh_value) else np.float64)
    ref_value = np.asarray(ref_value, dtype=np.complex128 if np.iscomplexobj(ref_value) else np.float64)
    assert fresh_value.shape == ref_value.shape, label

    peak = np.max(np.abs(ref_value))
    atol = max(peak * ATOL_FRACTION_OF_PEAK, 1e-9)

    with np.errstate(divide="ignore"):
        ref_db = 20 * np.log10(np.abs(ref_value) / peak)
    rtol = np.where(ref_db < NOISE_FLOOR_DB, LOOSE_RTOL, TIGHT_RTOL)

    diff = np.abs(fresh_value - ref_value)
    allowed = atol + rtol * np.abs(ref_value)
    bad = diff > allowed

    if bad.any():
        worst = np.unravel_index(np.argmax(diff - allowed), diff.shape)
        raise AssertionError(
            f"{label}: {bad.sum()}/{bad.size} elements exceed tolerance "
            f"(no longer matches the golden reference); worst at {worst}: "
            f"fresh={fresh_value[worst]!r} ref={ref_value[worst]!r} "
            f"diff={diff[worst]:.3g} allowed={allowed[worst]:.3g} "
            f"({ref_db[worst]:.1f} dB relative to peak)"
        )


@pytest.mark.parametrize("scenario_name", sorted(SCENARIOS))
def test_matches_golden_reference(scenario_name):
    reference = _load_reference(scenario_name)
    fresh = SCENARIOS[scenario_name]()

    assert set(fresh) == set(reference), "scenario outputs changed - regenerate golden data"

    for key, fresh_value in fresh.items():
        _assert_matches_reference(fresh_value, reference[key], f"{scenario_name}.{key}")
