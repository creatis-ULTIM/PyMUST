import numpy as np
from . import utils


# Documentation generated with Claude (claude-sonnet-4-6)
def getparam(probe: str) -> utils.Param:
    """Return parameters for a named ultrasound transducer array.

    Parameters
    ----------
    probe : str
        Transducer name (case-insensitive). Supported probes:

        Verasonics transducers:
          - ``'L11-5v'``  — 128-element, 7.6 MHz linear array
          - ``'L12-3v'``  — 192-element, 7.5 MHz linear array
          - ``'C5-2v'``   — 128-element, 3.6 MHz convex array
          - ``'P4-2v'``   — 64-element,  2.7 MHz phased array

        Other transducers:
          - ``'PA4-2/20'``  — 64-element,  2.5 MHz phased array
          - ``'L9-4/38'``   — 128-element, 5.0 MHz linear array
          - ``'LA530'``     — 192-element, 3.0 MHz linear array
          - ``'L14-5/38'``  — 128-element, 7.2 MHz linear array
          - ``'L14-5W/60'`` — 128-element, 7.5 MHz linear array
          - ``'P6-3'``      — 64-element,  4.5 MHz phased array

    Returns
    -------
    utils.Param
        Transducer parameter object with the following attributes:

        - ``Nelements`` : number of elements in the array
        - ``fc``        : center frequency (Hz)
        - ``pitch``     : element pitch (m)
        - ``width``     : element width (m)  *(when available)*
        - ``kerf``      : kerf width (m)
        - ``bandwidth`` : 6-dB fractional bandwidth (%)  *(when available)*
        - ``radius``    : radius of curvature (m); ``np.inf`` for linear arrays  *(when available)*
        - ``focus``     : elevation focus depth (m)  *(when available)*
        - ``height``    : element height (m)  *(when available)*

    Raises
    ------
    Exception
        If ``probe`` does not match any known transducer name.

    Examples
    --------
    Generate a focused pressure field with a phased-array transducer:

    >>> import numpy as np
    >>> from pymust import getparam, txdelay, pfield
    >>> param = getparam('P4-2v')          # 2.7 MHz phased array
    >>> x0, z0 = 2e-2, 5e-2               # focus position
    >>> dels = txdelay(x0, z0, param)      # TX time delays
    >>> x = np.linspace(-4e-2, 4e-2, 200)
    >>> z = np.linspace(param.pitch, 10e-2, 200)
    >>> xx, zz = np.meshgrid(x, z)
    >>> P = pfield(xx, np.zeros_like(xx), zz, dels, param)

    Notes
    -----
    Translated from the MATLAB MUST toolbox (Damien Garcia, 2015–2020).
    Hardware parameters for Verasonics probes are sourced from
    ``computeTrans.m`` (Verasonics, post-Aug 2019 version).

    See Also
    --------
    txdelay, pfield, simus, getpulse
    """
    param = utils.Param()
    probe = probe.upper()


    # from computeTrans.m (Verasonics, version post Aug 2019)
    if 'L11-5V' == probe:
        # --- L11-5v (Verasonics) ---
        param.fc = 7600000.0
        param.kerf = 3e-05
        param.width = 0.00027
        param.pitch = 0.0003
        param.Nelements = 128
        param.bandwidth = 77
        param.radius = np.inf
        param.height = 0.005
        param.focus = 0.018
    elif 'L12-3V' == probe:
        # --- L12-3v (Verasonics) ---
        param.fc = 7540000.0
        param.kerf = 3e-05
        param.width = 0.00017
        param.pitch = 0.0002
        param.Nelements = 192
        param.bandwidth = 93
        param.radius = np.inf
        param.height = 0.005
        param.focus = 0.02
    elif 'C5-2V' == probe:
        # --- C5-2v (Verasonics) ---
        param.fc = 3570000.0
        param.kerf = 4.8e-05
        param.width = 0.00046
        param.pitch = 0.000508
        param.Nelements = 128
        param.bandwidth = 79
        param.radius = 0.04957
        param.height = 0.0135
        param.focus = 0.06
    elif 'P4-2V' == probe:
        # --- P4-2v (Verasonics) ---
        param.fc = 2720000.0
        param.kerf = 5e-05
        param.width = 0.00025
        param.pitch = 0.0003
        param.Nelements = 64
        param.bandwidth = 74
        param.radius = np.inf
        param.height = 0.014
        param.focus = 0.06
        #--- From the OLD version of GETPARAM: ---#
    elif 'PA4-2/20' == probe:
        # --- PA4-2/20 ---
        param.fc = 2500000.0
        param.kerf = 5e-05
        param.pitch = 0.0003
        param.height = 0.014
        param.Nelements = 64
        param.bandwidth = 60
    elif 'L9-4/38' == (probe):
        # --- L9-4/38 ---
        param.fc = 5000000.0
        param.kerf = 3.5e-05
        param.pitch = 0.0003048
        param.height = 0.006
        param.Nelements = 128
        param.bandwidth = 65
    elif 'LA530' == (probe):
        # --- LA530 ---
        param.fc = 3000000.0
        width = 0.215 / 1000
        param.kerf = 0.03 / 1000
        param.pitch = width + param.kerf
        # element_height = 6/1000; # Height of element [m]
        param.Nelements = 192
    elif 'L14-5/38' == (probe):
        # --- L14-5/38 ---
        param.fc = 7200000.0
        param.kerf = 2.5e-05
        param.pitch = 0.0003048
        # height = 4e-3; # Height of element [m]
        param.Nelements = 128
        param.bandwidth = 70
    elif 'L14-5W/60' == (probe):
        # --- L14-5W/60 ---
        param.fc = 7500000.0
        param.kerf = 2.5e-05
        param.pitch = 0.000472
        # height = 4e-3; # Height of element [m]
        param.Nelements = 128
        param.bandwidth = 65
    elif 'P6-3' == (probe):
        # --- P6-3 ---
        param.fc = 4500000.0
        param.kerf = 2.5e-05
        param.pitch = 0.000218
        param.Nelements = 64
        param.bandwidth = 2 / 3 * 100
    else:
        raise Exception(np.array(['The probe ',probe,' is unknown. Should be one of [L11-5V, L12-3V, C5-2V, P4-2V, PA4-2/20, L9-4/38, LA530, L14-5/38, L14-5W/60, P6-3]']))

    return param
