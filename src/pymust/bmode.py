# Documentation generated with Claude (claude-sonnet-4-6)
import numpy as np
from . import utils
def bmode(IQ: np.ndarray, DR: float = 40) -> np.ndarray:
    """Convert I/Q signals to an 8-bit log-compressed B-mode image.

    Parameters
    ----------
    IQ : np.ndarray
        Complex I/Q array. Real part = in-phase component,
        imaginary part = quadrature component.
    DR : float, optional
        Dynamic range in dB (default 40 dB).
        If ``DR < 1``, treated as a gamma-compression exponent instead.

    Returns
    -------
    np.ndarray
        8-bit (uint8) log-compressed image with pixel values in [0, 255].

    Examples
    --------
    Demodulate RF signals and create B-mode images:

    >>> IQ = rf2iq(RF, param)
    >>> I = bmode(IQ, 30)

    Notes
    -----
    Translated from the MATLAB MUST toolbox (Damien Garcia, 2020).

    See Also
    --------
    rf2iq, tgc, sptrack
    """
    assert utils.iscomplex(IQ),'IQ must be a complex array'

    I = np.abs(IQ) # real envelope

    if (DR >= 1):
        I = 20*np.log10(I/np.max(I))+DR
        I = (255*I/DR) #.astype(np.uint8) # 8-bit log-compressed image
    else:    
        I = np.power(I / np.max(I), DR)
        I *= 255

    I[I<0] = 0
    I[I>255] = 255

    return I.astype(np.uint8)