import pymust


EXPECTED_FUNCTIONS = [
    "bmode",
    "dasmtx",
    "dasmtx3",
    "getparam",
    "impolgrid",
    "iq2doppler",
    "getNyquistVelocity",
    "pfield",
    "pfield3",
    "rf2iq",
    "simus",
    "simus3",
    "tgc",
    "txdelay",
    "txdelayCircular",
    "txdelayPlane",
    "txdelayFocused",
    "txdelay3",
    "txdelay3Plane",
    "txdelay3Diverging",
    "txdelay3Focused",
    "getDopplerColorMap",
    "genscat",
    "mkmovie",
    "getpulse",
    "smoothn",
    "sptrack",
]


def test_all_public_functions_are_importable():
    for name in EXPECTED_FUNCTIONS:
        assert hasattr(pymust, name), f"pymust.{name} is missing"
        assert callable(getattr(pymust, name))
