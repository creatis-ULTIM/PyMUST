import numpy as np
from . import utils    

def getpulse(param: utils.Param, way :int = 2, PreVel : str = 'pressure', dt : float = 1e-09): 
    #GETPULSE   Get the transmit pulse
#   PULSE = GETPULSE(PARAM,WAY) returns the one-way or two-way transmit
#   pulse with a time sampling of 1 nanosecond. Use WAY = 1 to get the
#   one-way pulse, or WAY = 2 to obtain the two-way (pulse-echo) pulse.
    
    #   PULSE = GETPULSE(PARAM,WAY,PRESVEL) returns the pulse in terms of
#   Pressure or Velocity. PRESVEL can be:
#       'pressure', which is the default
#       'velocity3D' or 'velocity2D'
    
    #   PULSE = GETPULSE(PARAM) uses WAY = 1 and PRESVEL = 'pressure'.
    
    #   [PULSE,t] = GETPULSE(...) also returns the time vector.
    
    #   PARAM is a structure which must contain the following fields:
#   ------------------------------------------------------------
#   1) PARAM.fc: central frequency (in Hz, REQUIRED)
#   2) PARAM.bandwidth: pulse-echo 6dB fractional bandwidth (in #)
#            The default is 75#.
#   3) PARAM.TXnow: number of wavelengths of the TX pulse (default: 1)
#   4) PARAM.TXfreqsweep: frequency sweep for a linear chirp (default: [])
#                         To be used to simulate a linear TX chirp.
    
    #   Example #1:
#   ----------
#   #-- Get the one-way pulse of a phased-array probe
#   # Phased-array @ 2.7 MHz:
#   param = getparam('P4-2v');
#   # One-way transmit pulse
#   [pulse,t] = getpulse(param);
#   # Plot the pulse
#   plot(t*1e6,pulse)
#   xlabel('{\mu}s')
#   axis tight
    
    #   Example #2:
#   ----------
#   #-- Check the pulse with a linear chirp
#   # Linear array:
#   param = getparam('L11-5v');
#   # Modify the fractional bandwidth:
#   param.bandwidth = 120;
#   # Define the properties of the chirp
#   param.TXnow = 20;
#   param.TXfreqsweep = 10e6;
#   # One-way transmit pulse
#   [pulse,t] = getpulse(param);
#   # Plot the pulse
#   plot(t*1e6,pulse)
#   xlabel('{\mu}s')
#   axis tight
    
    
    #   This function is part of <a
#   href="matlab:web('https://www.biomecardio.com/MUST')">MUST</a> (Matlab UltraSound Toolbox).
#   MUST (c) 2020 Damien Garcia, LGPL-3.0-or-later
    
    #   See also PFIELD, SIMUS, GETPARAM.
    
    #   -- Damien Garcia -- 2020/12, last update: 2021/05/26
#   website: <a
#   href="matlab:web('http://www.biomecardio.com')">www.BiomeCardio.com</a>
    
    
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

    # Corrected frequencies
    P = np.abs(F)**2
    Fc = np.trapz(f*P) / np.trapz(P)
    f = f + Fc - fc

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
    
    
