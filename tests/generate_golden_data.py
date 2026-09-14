"""Regenerate the golden reference .npz files used by test_regression.py.

Run this deliberately after a change that intentionally alters simus/pfield
numerics, then review the diff of the resulting .npz files:

    python tests/generate_golden_data.py
"""
import numpy as np

from golden_scenarios import DATA_DIR, SCENARIOS


def main():
    DATA_DIR.mkdir(exist_ok=True)
    for name, build in SCENARIOS.items():
        outputs = build()
        path = DATA_DIR / f"{name}.npz"
        np.savez(path, **outputs)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
