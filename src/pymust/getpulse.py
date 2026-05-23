"""
.. note:: Documentation auto-generated with Claude (claude-sonnet-4-6).
"""
from __future__ import annotations
import numpy as np
from . import utils

def getpulse(param: utils.Param, way: int = 2, PreVel: str = 'pressure', dt: float = 1e-09) -> tuple[np.ndarray, np.ndarray]:
    """Return the transmit pulse waveform for a given transducer.

    Parameters
    ----------
    param : utils.Param
        Transducer parameter structure with the following fields:

        - ``fc``          : center frequency (Hz, required)
        - ``bandwidth``   : pulse-echo 6 dB fractional bandwidth (%, default 75)
        - ``TXnow``       : TX pulse length in wavelengths (default 1)
        - ``TXfreqsweep`` : frequency sweep for a linear chirp (Hz, default None)
    way : int, optional
        ``1`` for one-way pulse, ``2`` for two-way pulse-echo (default 2).
    PreVel : str, optional
        Quantity to return: ``'pressure'`` (default), ``'velocity2D'``, or
        ``'velocity3D'``.
    dt : float, optional
        Time sampling step (s, default 1 ns).

    Returns
    -------
    pulse : np.ndarray
        Normalised pulse waveform (peak amplitude = 1).
    t : np.ndarray
        Corresponding time vector (s).

    Examples
    --------
    One-way pulse of a phased-array probe:

    >>> param = getparam('P4-2v')
    >>> pulse, t = getpulse(param, way=1)

    Linear chirp pulse:

    >>> param = getparam('L11-5v')
    >>> param.bandwidth = 120
    >>> param.TXnow = 20
    >>> param.TXfreqsweep = 10e6
    >>> pulse, t = getpulse(param, way=1)

    Notes
    -----
    Translated from the MATLAB MUST toolbox (Damien Garcia, 2020–2021).

    See Also
    --------
    pfield, simus, getparam
    """

    assert way == 1 or way == 2,'WAY must be 1 (one-way) or 2 (two-way)'
    #-- Check PreVel
    PreVelValid = ['pressure','pres','velocity2d','velocity3d','vel2d','vel3d']
    assert PreVel.lower() in PreVelValid, 'PRESVEL must be one of: ' + ', '.joiN(PreVelValid)
    #-- Center frequency (in Hz)
    assert 'fc' in param,'A center frequency value (PARAM.fc) is required.'
    fc = param.fc
    
    #-- Fractional bandwidth at -6dB (in #)
    if not 'bandwidth' in param :
        param.bandwidth = 75
    
    assert param.bandwidth > 0 and param.bandwidth < 200,'The fractional bandwidth at -6 dB (PARAM.bandwidth, in %) must be in ]0,200['

    #-- TX pulse: Number of wavelengths
    if 'TXnow' not in param :
        param.TXnow = 1
    
    NoW = param.TXnow
    
    assert np.isscalar(NoW) and utils.isnumeric(NoW) and NoW > 0,'PARAM.TXnow must be a positive scalar.'

    #-- TX pulse: Frequency sweep for a linear chirp
    if not 'TXfreqsweep' in param or param.TXfreqsweep is None or np.isinf(param.TXfreqsweep):
        param.TXfreqsweep = None
    
    FreqSweep = param.TXfreqsweep
    assert FreqSweep is None or (np.isscalar(FreqSweep) and utils.isnumeric(FreqSweep) and FreqSweep > 0),'PARAM.TXfreqsweep must be None (windowed sine) or a positive scalar (linear chirp).'
    
    pulseSpectrum = param.getPulseSpectrumFunction(FreqSweep)
    
    #-- FREQUENCY RESPONSE of the ensemble PZT + probe
    probeSpectrum = param.getProbeFunction()
    # Note: The spectrum of the pulse (pulseSpectrum) will be then multiplied
    # by the frequency-domain tapering window of the transducer (probeSpectrum)
    
    #-- frequency samples
    eps = 1e-9 

    df = param.fc / param.TXnow / 32
    p = utils.nextpow2(1 / dt / 2 / df)
    Nf = 2 ** p
    f = np.linspace(0,1 / dt / 2,Nf)
    #-- spectrum of the pulse
    F = np.multiply(pulseSpectrum(2 * np.pi * f),probeSpectrum(2 * np.pi * f) ** way)
    if  PreVel.lower() in ['vel2d','velocity2d']:
        F = F / (np.sqrt(f) + eps)
    elif PreVel.lower() in ['vel3d','velocity3d']:
            F = F / (f + eps)

    # Corrected frequencies % (added on April 24, 2023 ; removed on Nov 8, 2023) on MUST
    # P = np.abs(F)**2
    # Fc = np.trapz(f*P) / np.trapz(P)
    # f = f + Fc - fc

    F = np.multiply(pulseSpectrum(2 * np.pi * f),probeSpectrum(2 * np.pi * f) ** way)
    
    #-- pulse in the temporal domain (step = 1 ns)
    pulse = np.fft.fftshift(np.fft.irfft(F))
    pulse = pulse / np.max(np.abs(pulse))
    #-- keep the significant magnitudes
    idx, = np.where(pulse > (1 / 1023))
    idx1 = idx[0]
    idx2 = idx[-1]
    idx = min(idx1 + 1, 2 * Nf - 1 - idx2-1)
    #pulse = pulse[np.arange(end() - idx + 1,idx+- 1,- 1)
    pulse = pulse[-idx: idx-2:-1]
    #-- time vector
    t = np.arange(len(pulse)) *dt
    return pulse,t
    
    
