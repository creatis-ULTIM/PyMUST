"""
.. note:: Documentation auto-generated with Claude (claude-sonnet-4-6).
"""
import numpy as np, scipy, scipy.signal, logging
from . import utils
from typing import Union

def rf2iq(RF: np.ndarray, Fs: Union[float, utils.Param], Fc: float = None, B: float = None) -> np.ndarray:
    """Demodulate RF signals to complex I/Q representation.

    Down-mixes bandpass RF signals to baseband and low-pass filters them to
    return complex I/Q (in-phase / quadrature) data.

    Parameters
    ----------
    RF : np.ndarray
        Real-valued RF signals.  Each column is one signal over fast-time.
    Fs : float or utils.Param
        Sampling frequency of the RF signals (Hz).  Alternatively, pass a
        :class:`~pymust.utils.Param` structure with the following fields:

        - ``fs``        : sampling frequency (Hz, required)
        - ``fc``        : center frequency (Hz, optional; required for
          undersampled signals)
        - ``bandwidth`` : fractional bandwidth (%, optional)
        - ``t0``        : time offset (s, optional, default 0)
    Fc : float, optional
        Carrier (center) frequency (Hz).  If ``None`` (default), it is
        estimated from the power spectrum of *RF*.  **Must be provided for
        undersampled (bandpass-sampled) signals.**
    B : float, optional
        Fractional bandwidth in % (``B = bandwidth_Hz * 100 / Fc``).  When
        given, the low-pass cut-off is ``Wn = Fc * B / 100 / Fs`` instead of
        the default ``min(2*Fc/Fs, 0.5)``.

    Returns
    -------
    IQ : np.ndarray
        Complex I/Q array with the same shape as *RF*. ``real(IQ)`` is the
        in-phase component; ``imag(IQ)`` is the quadrature component.

    Notes
    -----
    Method: multiply *RF* by ``exp(-j*2*pi*Fc*t)`` (down-mixing), then apply
    a 5th-order Butterworth low-pass filter with ``scipy.signal.filtfilt``,
    and multiply by 2 to restore the envelope amplitude.

    A warning is emitted if harmful aliasing is detected (undersampled case).

    Translated from the MATLAB MUST toolbox (Damien Garcia, 2012–2020).
    MUST (c) 2020 Damien Garcia, LGPL-3.0-or-later

    References
    ----------
    Madiena C, Faurie J, Porée J, Garcia D. Color and vector flow imaging in
    parallel ultrasound with sub-Nyquist sampling. IEEE Trans Ultrason
    Ferroelectr Freq Control, 2018;65:795-802.

    See Also
    --------
    iq2doppler, bmode, wfilt
    """

    #%-- Check input arguments
    assert np.issubdtype(RF.dtype, np.floating),'RF must contain real RF signals.'
    t0 = 0; #% default value for time offset
    if isinstance(Fs, utils.Param):
        param = Fs
        param.ignoreCaseInFieldNames()
        assert utils.isfield(param,'fs'), 'A sampling frequency (PARAM.fs) is required.'
        Fs = param.fs
        B = param.get('bandwidth', None)
        Fc = param.get('fc', None)
        t0 = param.get('t0', np.zeros((1)))


    assert np.isscalar(Fs), 'The sampling frequency (Fs or PARAM.fs) must be a scalar.'
    assert Fc is None or np.isscalar(Fc), 'The center frequency (Fc or PARAM.fc) must be None or a scalar.'

    #%-- Convert to column vector (if RF is a row vector)

    #%-- Time vector
    nl = RF.shape[0]
    t = np.arange(nl)/Fs
    if isinstance(t0, float):
        t0 = np.ones((1))*t0 
    assert utils.isnumeric(t0) and np.isscalar(t0) or isinstance(t0, np.ndarray) and (len(t0)==1 or len(t0)==nl), 'PARAM.t0 must be a numeric scalar or vector of size = size(RF,1).'
    t = t+t0

    #%-- Seek the carrier frequency (if required)
    if Fc is None:
        #% Keep a maximum of 100 randomly selected scanlines
        Nc = RF.shape[1]
        if Nc<100:
             idx = np.arange(Nc)
        else:
            idx = np.random.permutation(Nc)[:100]
        #% Power Spectrum
        P = np.linalg.norm(np.fft.rfft(RF[:,idx], axis = 0),axis =1)
        freqs = np.fft.rfftfreq(RF.shape[0],1/Fs)
        #% Carrier frequency
        Fc = np.sum(freqs*P)/np.sum(P)
    
    #%-- Normalized cut-off frequency
    if B is None:
        Wn = min(2*Fc/Fs,0.5)
    else:
        assert np.isscalar(B), 'The signal bandwidth (B or PARAM.bandwidth) must be a scalar.'
        assert B>0 and B<200, 'The signal bandwidth (B or PARAM.bandwidth, in %) must be within the interval of ]0,200[.'
        B = Fc*B/100 #; % bandwidth in Hz
        Wn = B/Fs

    assert Wn>0 and Wn<=1,'The normalized cutoff frequency is not within the interval of (0,1). Check the input parameters!'

    #%-- Down-mixing of the RF signals
    exponential = np.exp(-1j*2*np.pi*Fc*t)
    exponential = exponential.reshape( [-1] + [1 for _ in range(RF.ndim-1)])
    IQ =exponential*RF


   # %-- Low-pass filter
    b,a = scipy.signal.butter(5,Wn)
    IQ = scipy.signal.filtfilt(b,a,IQ, axis = 0)*2; #% factor 2: to preserve the envelope amplitude

    #%-- Recover the initial size (if was a vector row)
    #if wasrow:
    #      IQ = IQ.T # end

    #%-- Display a warning message if harmful aliasing is suspected
    if B is not None and Fs<(2*Fc+B): #% the RF signal is undersampled
        fL = Fc-B/2; fH = Fc+B/2; #% lower and higher frequencies of the bandpass signal
        n = int(np.floor(fH/(fH-fL)))
        harmlessAliasing = np.any(np.logical_and(2*fH/np.arange(1,n+1) <=Fs,  Fs<=2*fL/(np.arange(n) +1e-10)))
        if not harmlessAliasing:
            logging.warning('RF2IQ:harmfulAliasing: Harmful aliasing is present: the aliases are not mutually exclusive!')
    return IQ