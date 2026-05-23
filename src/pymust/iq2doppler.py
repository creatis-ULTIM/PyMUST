"""
.. note:: Documentation auto-generated with Claude (claude-sonnet-4-6).
"""
from __future__ import annotations
import numpy as np,scipy, scipy.signal, typing
from . import utils

def iq2doppler(IQ: np.ndarray, param: utils.Param, M: typing.Union[int,np.ndarray] = 1, lag: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Convert I/Q data to color Doppler velocity and variance.

    Estimates Doppler velocity using a slow-time autocorrelator (Kasai
    estimator) on a 3-D complex I/Q array.

    Parameters
    ----------
    IQ : np.ndarray
        3-D complex array of shape ``(depth, elements, slow_time)``.
        The real and imaginary parts are the in-phase and quadrature
        components, respectively.  The full ensemble (3rd dimension) is used.
    param : utils.Param
        Parameter structure with the following fields:

        - ``fc``  : center frequency (Hz, required)
        - ``c``   : speed of sound (m/s, default 1540)
        - ``PRF`` : pulse repetition frequency (Hz) **or**
        - ``PRP`` : pulse repetition period (s) — one of the two is required
    M : int or array-like, optional
        Spatial averaging window:

        - scalar *M* → M×M neighbourhood (default 1, no averaging)
        - 2-element vector ``[M1, M2]`` → M1×M2 neighbourhood
    lag : int, optional
        Autocorrelation lag (default 1).

    Returns
    -------
    vel : np.ndarray
        Doppler velocity estimate (m/s), same spatial shape as ``IQ[:,:,0]``.
    variance : np.ndarray
        Doppler variance estimate (m²/s²), same shape as *vel*.

    Notes
    -----
    Based on Eq. 55 in Loupas et al. (IEEE UFFC 42, 4; 1995).
    Translated from the MATLAB MUST toolbox (Damien Garcia & Jonathan Porée,
    2015–2020). MUST (c) 2020 Damien Garcia, LGPL-3.0-or-later

    References
    ----------
    Madiena C, Faurie J, Porée J, Garcia D. Color and vector flow imaging in
    parallel ultrasound with sub-Nyquist sampling. IEEE Trans Ultrason
    Ferroelectr Freq Control, 2018;65:795-802.

    See Also
    --------
    rf2iq, wfilt, getNyquistVelocity
    """

    if isinstance(M, int):
        M = M *np.ones(2, dtype = int)

    assert np.all(M>0) and M.dtype == int, 'M must contain integers >0'
    #%-
    if len(IQ.shape)==4:
        raise ValueError('IQ is a 4-D array: use IQ2DOPPLER3.')

    assert len(IQ.shape)==3,'IQ must be a 3-D array'
    #%-

    assert isinstance(lag,int) and lag>0, 'The 4th input parameter LAG must be a positive integer'

    #%----- Input parameters in PARAM -----
    #%-- 1) Speed of sound
    assert isinstance(param, utils.Param), 'param should be a Param class'
    if not utils. isfield(param,'c'):
        param.c = 1540 # % longitudinal velocity in m/s

    c = param.c
    #%-- 2) Center frequency
    if utils.isfield(param,'fc'):
        fc = param.fc
    else:
        raise ValueError('A center frequency (fc) must be specified in the structure PARAM: PARAM.fc')

    #%-- 3) Pulse repetition frequency or period (PRF or PRP)
    if utils.isfield(param,'PRF'):
        PRF = param.PRF
    elif utils.isfield(param,'PRP'):
        PRF = 1./param.PRP
    else:
        raise ValueError('A pulse repetition frequency or period must be specified in the structure PARAM: PARAM.PRF or PARAM.PRP')

    if utils.isfield(param,'PRP') and utils.isfield(param,'PRF'):
        assert abs(param.PRF-1./param.PRP)<utils.eps(), 'A conflict exists for the pulse repetition frequency & period: PARAM.PRF and 1/PARAM.PRP are different!'



    #%--- AUTO-CORRELATION METHOD ---
    #% Eq. 55 in Loupas et al. (IEEE UFFC 42,4;1995)
    IQ1 = IQ[:,:,: -lag]
    IQ2 = IQ[:,:,lag:]

    AC = np.sum(IQ1*np.conj(IQ2),2)  # ensemble auto-correlation

    if  M[0] != 1 or M[1] != 1: #isequal([M(1) M(2)],[1 1]) % spatial weighted average
        h = scipy.signal.windows.hamming(M[0]).reshape((-1, 1))*scipy.signal.windows.hamming(M[1]).reshape((1, -1))
        AC = scipy.signal.convolve2d(AC,h, 'same', boundary='symmetric')

    #%-- Doppler velocity
    VN = c*PRF/4/fc/lag; #% Nyquist velocity
    vel = -VN*(np.angle(AC))/np.pi; 


    #%-- Doppler variance
    P = np.sum((IQ.real)**2+(IQ.imag)**2,2) #% power
    if  M[0] != 1 or M[1] != 1: # % spatial weighted average
        P = scipy.signal.convolve2d(P,h, 'same', boundary='symmetric')

    variance = 2*(VN/np.pi)**2*(1-np.abs(AC)/P)
    #%-- cf. Eq. 7.48 in Estimation of Blood Velocities Using Ultrasound:
    #%   A Signal Processing Approach by Jørgen Arendt Jensen,
    #%   Cambridge University Press, 1996
    return vel, variance

def getNyquistVelocity(param, lag=  1):
    if not utils. isfield(param,'c'):
        param.c = 1540 # % longitudinal velocity in m/s

    if utils.isfield(param,'PRF'):
        PRF = param.PRF
    elif utils.isfield(param,'PRP'):
        PRF = 1./param.PRP
    else:
        raise ValueError('A pulse repetition frequency or period must be specified in the structure PARAM: PARAM.PRF or PARAM.PRP')

    return  param.c*PRF/4/param.fc/lag






