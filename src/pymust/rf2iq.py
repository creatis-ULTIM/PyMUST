import numpy as np, scipy, scipy.signal, logging
from . import utils
from typing import Union

def rf2iq(RF : np.ndarray, Fs : Union[float, utils.Param], Fc : float = None, B : float = None):
    """
    %RF2IQ   I/Q demodulation of RF data
    %   IQ = RF2IQ(RF,Fs,Fc) demodulates the radiofrequency (RF) bandpass
    %   signals and returns the Inphase/Quadrature (I/Q) components. IQ is a
    %   complex whose real (imaginary) part contains the inphase (quadrature)
    %   component.
    %       1) Fs is the sampling frequency of the RF signals (in Hz),
    %       2) Fc represents the center frequency (in Hz).
    %
    %   IQ = RF2IQ(RF,Fs) or IQ = RF2IQ(RF,Fs,[],...) calculates the carrier
    %   frequency.
    %   IMPORTANT: Fc must be given if the RF signal is undersampled (as in
    %   bandpass sampling).
    %
    %   [IQ,Fc] = RF2IQ(...) also returns the carrier frequency (in Hz).
    %
    %   RF2IQ uses a downmixing process followed by low-pass filtering. The
    %   low-pass filter is determined by the normalized cut-off frequency Wn.
    %   By default Wn = min(2*Fc/Fs,0.5). The cut-off frequency Wn can be
    %   adjusted if the relative bandwidth (in %) is given:
    %
    %   IQ = RF2IQ(RF,Fs,Fc,B)
    %
    %   The bandwidth in % is defined by:
    %     B = Bandwidth_in_% = Bandwidth_in_Hz*(100/Fc).
    %   When B is an input variable, the cut-off frequency is
    %     Wn = Bandwidth_in_Hz/Fs, i.e:
    %     Wn = B*(Fc/100)/Fs. 
    %       
    %   If there is a time offset, use PARAM.t0, as explained below.
    %
    %   An alternative syntax for RF2IQ is the following:
    %   IQ = RF2IQ(RF,PARAM), where the structure PARAM must contain the
    %   required parameters:
    %       1) PARAM.fs: sampling frequency (in Hz, REQUIRED)
    %       2) PARAM.fc: center frequency (in Hz, OPTIONAL, required for
    %            undersampled RF signals)
    %       3) PARAM.bandwidth: fractional bandwidth (in %, OPTIONAL)
    %       4) PARAM.t0: time offset (in s, OPTIONAL, default = 0)
    %
    %   Notes on Undersampling (sub-Nyquist sampling)
    %   ----------------------
    %   If the RF signal is undersampled, the carrier frequency Fc must be
    %   specified. If a fractional bandwidth (B or PARAM.bandwidth) is given, a
    %   warning message appears if harmful aliasing is suspected.
    %
    %   Notes:
    %   -----
    %   RF2IQ treats the data along the first non-singleton dimension as
    %   vectors, i.e. RF2IQ demodulates along columns for 2-D and 3-D RF data.
    %   Each column corresponds to a single RF signal over (fast-) time.
    %   Use IQ2RF to recover the RF signals.
    %
    %   Method:
    %   ------
    %   RF2IQ multiplies RF by a phasor of frequency Fc (down-mixing) and
    %   applies a fifth-order Butterworth lowpass filter using FILTFILT:
    %       IQ = RF.*exp(-1i*2*pi*Fc*t);
    %       [b,a] = butter(5,2*Fc/Fs);
    %       IQ = filtfilt(b,a,IQ)*2;
    %
    %
    %   Example #1: Envelope of an RF signal
    %   ----------
    %   % Load an RF signal sampled at 20 MHz
    %   load RFsignal@20MHz.mat
    %   % I/Q demodulation
    %   IQ = rf2iq(RF,20e6);
    %   % RF signal and its envelope
    %   plot(RF), hold on
    %   plot(abs(IQ),'Linewidth',1.5), hold off
    %   legend({'RF signal','I/Q amplitude'})
    %
    %   Example #2: Demodulation of an undersampled RF signal
    %   ----------
    %   % Load an RF signal sampled at 20 MHz
    %   % (Center frequency = 5 MHz / Bandwidth = 2 MHz)
    %   load RFsignal@20MHz.mat
    %   % I/Q demodulation of the original RF signal
    %   Fs = 20e6;
    %   IQ = rf2iq(RF,Fs);
    %   % Create an undersampled RF signal (sampling at Fs/5 = 4 MHz)
    %   bpsRF = RF(1:5:end);
    %   subplot(211), plot(1:1000,RF,1:5:1000,bpsRF,'.-')
    %   title('RF signal (5 MHz array)')
    %   legend({'sampled @ 20 MHz','bandpass sampled @ 4 MHz'})
    %   % I/Q demodulation of the undersampled RF signal
    %   Fs = 4e6; Fc = 5e6;
    %   iq = rf2iq(bpsRF,Fs,Fc);
    %   % Display the IQ signals
    %   subplot(212), plot(1:1000,abs(IQ),1:5:1000,abs(iq),'.-')
    %   title('I/Q amplitude')
    %   legend({'sampled @ 20 MHz','bandpass sampled @ 4 MHz'})
    %
    %
    %   REFERENCE
    %   ---------
    %   If you find this function useful, you can cite the following paper.
    %
    %   1) Madiena C, Faurie J, Porée J, Garcia D, Color and vector flow
    %   imaging in parallel ultrasound with sub-Nyquist sampling. IEEE Trans
    %   Ultrason Ferroelectr Freq Control, 2018;65:795-802.
    %   <a
    %   href="matlab:web('http://www.biomecardio.com/publis/ieeeuffc18a.pdf')">download PDF</a>
    %
    %
    %   This function is part of MUST (Matlab UltraSound Toolbox).
    %   MUST (c) 2020 Damien Garcia, LGPL-3.0-or-later
    %
    %   See also IQ2DOPPLER, IQ2RF, BMODE, WFILT.
    %
    %   -- Damien Garcia -- 2012/01, last update: 2020/05
    %   website: <a
    %   href="matlab:web('https://www.biomecardio.com')">www.BiomeCardio.com</a>
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