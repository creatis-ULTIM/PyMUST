import warnings
import numpy as np
from . import utils


def _isscalarlike(x):
    """True for a plain scalar or a size-1 array - several PyMUST functions
    stash what is conceptually a scalar as a (1,) or (1,1) array (a MATLAB
    row/column-vector convention), e.g. PARAM.t0 or PARAM.RXangle."""
    return np.isscalar(x) or (isinstance(x, np.ndarray) and x.size == 1)


class MediumParams:
    """Acoustic medium properties (PARAM.medium)."""
    def __init__(self):
        self.c = None            # speed of sound (m/s), default 1540
        self.attenuation = None  # attenuation coefficient (dB/cm/MHz), default 0
        self.rho = None          # density (kg/m^3), default 1050
        self.beta = None         # nonlinearity parameter, default 4.5 (soft tissue)

    def check(self):
        if self.c is None:
            self.c = 1540
        assert utils.isnumeric(self.c) and _isscalarlike(self.c) and self.c > 0, \
            'PARAM.medium.c must be a positive scalar.'

        if self.attenuation is None:
            self.attenuation = 0
        assert utils.isnumeric(self.attenuation) and _isscalarlike(self.attenuation) and self.attenuation >= 0, \
            'PARAM.medium.attenuation must be a nonnegative scalar.'

        if self.rho is None:
            self.rho = 1050
        assert utils.isnumeric(self.rho) and _isscalarlike(self.rho) and self.rho > 0, \
            'PARAM.medium.rho must be a positive scalar.'

        if self.beta is None:
            self.beta = 4.5
        assert utils.isnumeric(self.beta) and _isscalarlike(self.beta) and self.beta >= 0, \
            'PARAM.medium.beta must be a nonnegative scalar.'
        return self


class XdcrParams:
    """Transducer geometry and properties (PARAM.xdcr)."""
    def __init__(self):
        self.fc = None
        self.pitch = None
        self.width = None
        self.kerf = None
        self.bandwidth = None
        self.radius = None
        self.focus = None
        self.height = None
        self.nelements = None
        self.baffle = None
        self.elements = None  # 2-row (x,y) array of element centers, for matrix arrays

    @property
    def non_rigid_baffle(self):
        """Whether an obliquity factor is needed (baffle is anything but 'rigid')."""
        return self.baffle != 'rigid'

    def check(self):
        if self.fc is not None:
            assert utils.isnumeric(self.fc) and _isscalarlike(self.fc) and self.fc > 0, \
                'The center frequency (PARAM.xdcr.fc) must be positive.'

        if self.nelements is not None:
            assert utils.isnumeric(self.nelements) and _isscalarlike(self.nelements) and self.nelements > 0 \
                and self.nelements == round(self.nelements), \
                'The number of elements (PARAM.xdcr.nelements) is invalid.'

        if self.bandwidth is None:
            self.bandwidth = 75
        assert utils.isnumeric(self.bandwidth) and _isscalarlike(self.bandwidth) and 0 < self.bandwidth < 200, \
            'The fractional bandwidth (PARAM.xdcr.bandwidth) must be in ]0,200[.'

        if self.focus is None:
            self.focus = np.inf
        assert utils.isnumeric(self.focus) and _isscalarlike(self.focus) and self.focus > 0, \
            'The elevation focus (PARAM.xdcr.focus) must be positive.'

        if self.height is None:
            self.height = np.inf
        assert utils.isnumeric(self.height) and _isscalarlike(self.height) and self.height > 0, \
            'The element height (PARAM.xdcr.height) must be positive.'

        if self.radius is None:
            self.radius = np.inf
        assert utils.isnumeric(self.radius) and _isscalarlike(self.radius) and self.radius > 0, \
            'The radius of curvature (PARAM.xdcr.radius) must be positive.'

        if self.baffle is None:
            self.baffle = 'soft'
        if isinstance(self.baffle, str):
            assert self.baffle.lower() in ('rigid', 'soft'), \
                "The baffle (PARAM.xdcr.baffle) must be 'rigid' or 'soft'."
        else:
            assert utils.isnumeric(self.baffle) and _isscalarlike(self.baffle) and self.baffle > 0, \
                'The baffle scalar (PARAM.xdcr.baffle) must be positive.'

        # Pitch, width, and kerf must be mutually consistent (pitch = width + kerf)
        if self.pitch is not None:
            assert utils.isnumeric(self.pitch) and _isscalarlike(self.pitch) and self.pitch > 0, \
                'The pitch (PARAM.xdcr.pitch) must be positive.'
            if self.width is not None and self.kerf is not None:
                tol = 10 * utils.eps() * max(abs(self.pitch), abs(self.width), abs(self.kerf), 1.0)
                assert abs(self.pitch - self.width - self.kerf) <= tol, \
                    'PARAM.xdcr.pitch must equal width + kerf.'
            elif self.kerf is not None:
                width = self.pitch - self.kerf
                assert width > 0, 'PARAM.xdcr.pitch must be greater than PARAM.xdcr.kerf.'
                self.width = width
            elif self.width is not None:
                kerf = self.pitch - self.width
                assert kerf >= 0, 'PARAM.xdcr.pitch must be greater than or equal to PARAM.xdcr.width.'
                self.kerf = kerf
        elif self.width is not None and self.kerf is not None:
            self.pitch = self.kerf + self.width

        if self.kerf is not None:
            assert utils.isnumeric(self.kerf) and _isscalarlike(self.kerf) and self.kerf >= 0, \
                'The kerf width (PARAM.xdcr.kerf) must be nonnegative.'
        if self.width is not None:
            assert utils.isnumeric(self.width) and _isscalarlike(self.width) and self.width > 0, \
                'The element width (PARAM.xdcr.width) must be positive.'

        # Coordinates of the transducer elements (for matrix arrays)
        if self.elements is not None:
            elements = np.asarray(self.elements)
            assert elements.ndim == 2 and elements.shape[0] == 2 and elements.shape[1] > 0, \
                ('PARAM.xdcr.elements must be a nonempty two-row numeric array containing '
                 'the x- and y-coordinates.')
            nElementsFromCoordinates = elements.shape[1]
            if self.nelements is not None:
                assert self.nelements == nElementsFromCoordinates, \
                    'PARAM.xdcr.nelements must equal the number of columns in PARAM.xdcr.elements.'
            else:
                self.nelements = nElementsFromCoordinates
        return self


class TxParams:
    """Transmit properties (PARAM.tx)."""
    def __init__(self):
        self.fe = None            # excitation frequency (Hz), default = xdcr.fc
        self.now = None           # number of wavelengths, default 1
        self.freqsweep = None     # linear chirp bandwidth (Hz); None = windowed sine
        self.apodization = None   # transmit apodization, default ones(nelements)
        self.delay = None         # transmit delays (s)
        self.passive = None       # passive (receive-only) imaging flag, default False
        self.prf = None           # pulse repetition frequency (Hz)
        self.prp = None           # pulse repetition period (s) = 1/prf

    def check(self, xdcr: 'XdcrParams'):
        if xdcr.fc is not None and self.fe is None:
            self.fe = xdcr.fc
        if self.fe is not None:
            assert utils.isnumeric(self.fe) and _isscalarlike(self.fe) and self.fe > 0, \
                'The excitation frequency (PARAM.tx.fe) must be positive.'
            if xdcr.fc is not None:
                assert abs(self.fe - xdcr.fc) < (xdcr.fc * xdcr.bandwidth / 200), \
                    "The excitation frequency (PARAM.tx.fe) is outside the transducer's bandwidth."

        if self.now is None:
            self.now = 1
        assert _isscalarlike(self.now) and utils.isnumeric(self.now) and self.now > 0, \
            'PARAM.tx.now must be a positive scalar.'

        # A frequency sweep (linear chirp) doesn't apply to an infinitely long pulse
        if np.isinf(self.now):
            self.freqsweep = None

        if self.freqsweep is not None:
            assert _isscalarlike(self.freqsweep) and utils.isnumeric(self.freqsweep) and self.freqsweep > 0, \
                'PARAM.tx.freqsweep must be None (windowed sine) or a positive scalar (linear chirp).'

        if xdcr.nelements is not None and self.apodization is None:
            self.apodization = np.ones((1, xdcr.nelements), dtype=np.float32)
        if self.apodization is not None:
            apod = np.atleast_2d(self.apodization)
            assert apod.ndim == 2 and utils.isnumeric(apod), 'PARAM.tx.apodization must be a numeric vector.'
            if xdcr.nelements is not None:
                assert apod.size == xdcr.nelements, \
                    'PARAM.tx.apodization must have length = (number of elements).'
            self.apodization = apod

        if self.delay is not None:
            delay = np.asarray(self.delay)
            assert delay.ndim <= 2 and utils.isnumeric(delay), \
                'PARAM.tx.delay must be a numeric vector or matrix.'
            if xdcr.nelements is not None:
                assert delay.shape[-1] == xdcr.nelements, \
                    'PARAM.tx.delay must be a row vector or a matrix whose columns correspond to elements.'

        if self.passive is None:
            self.passive = False
        assert isinstance(self.passive, (bool, np.bool_)) or (utils.isnumeric(self.passive) and self.passive in (0, 1)), \
            'PARAM.tx.passive must be True, False, 0, or 1.'
        self.passive = bool(self.passive)

        if self.prf is not None:
            assert utils.isnumeric(self.prf) and _isscalarlike(self.prf) and np.isfinite(self.prf) and self.prf > 0, \
                'PARAM.tx.prf must be a positive scalar.'
        if self.prp is not None:
            assert utils.isnumeric(self.prp) and _isscalarlike(self.prp) and np.isfinite(self.prp) and self.prp > 0, \
                'PARAM.tx.prp must be a positive scalar.'
        if self.prf is not None and self.prp is not None:
            tol = 10 * utils.eps() * max(abs(self.prf), abs(1 / self.prp), 1.0)
            assert abs(self.prf - 1 / self.prp) <= tol, 'PARAM.tx.prf must equal 1/PARAM.tx.prp.'
        elif self.prp is not None:
            self.prf = 1 / self.prp
        return self


class RxParams:
    """Receive properties (PARAM.rx)."""
    def __init__(self):
        self.fs = None            # sampling frequency (Hz), default = 4*xdcr.fc
        self.fnumber = None       # f-number for dynamic aperture, default 0 (all elements)
        self.delay = None         # receive delays (s), default zeros(nelements)
        self.angle = None         # receive angle (rad), default 0
        self.apodization = None   # receive apodization, default 'rectangular'
        self.t0 = None            # acquisition start time (s), default 0

    def check(self, xdcr: 'XdcrParams'):
        if xdcr.fc is not None and self.fs is None:
            self.fs = 4 * xdcr.fc
        if self.fs is not None:
            assert utils.isnumeric(self.fs) and _isscalarlike(self.fs) and self.fs > 0, \
                'PARAM.rx.fs must be a positive scalar.'

        if xdcr.nelements is not None and self.delay is None:
            self.delay = np.zeros((1, xdcr.nelements), dtype=np.float32)
        if self.delay is not None:
            delay = np.atleast_2d(self.delay)
            assert delay.ndim == 2 and utils.isnumeric(delay), 'PARAM.rx.delay must be a numeric vector.'
            if xdcr.nelements is not None:
                assert delay.size == xdcr.nelements, \
                    'PARAM.rx.delay must have length = (number of elements).'
            self.delay = delay

        if self.angle is None:
            self.angle = 0
        assert utils.isnumeric(self.angle) and _isscalarlike(self.angle), 'PARAM.rx.angle must be a numeric scalar.'

        if self.fnumber is None:
            self.fnumber = 0
        else:
            assert utils.isnumeric(self.fnumber) and _isscalarlike(self.fnumber) and self.fnumber >= 0, \
                'PARAM.rx.fnumber must be a nonnegative scalar.'

        if self.apodization is None:
            self.apodization = 'rectangular'
        else:
            assert isinstance(self.apodization, str), 'PARAM.rx.apodization must be a string.'
            assert self.apodization.lower() in \
                ('rectangular', 'boxcar', 'tukey', 'hann', 'hanning', 'hamming'), \
                ("PARAM.rx.apodization must be 'rectangular', 'boxcar', 'tukey', 'hann', "
                 "'hanning', or 'hamming'.")

        if self.t0 is None:
            self.t0 = 0
        assert utils.isnumeric(self.t0) and _isscalarlike(self.t0) and self.t0 >= 0, \
            'PARAM.rx.t0 must be a nonnegative scalar.'
        return self


class Param(utils.dotdict):
    """Transducer/medium/transmit/receive parameters used throughout PyMUST.

    As of this version, PARAM is internally organized into nested sub-objects
    (mirroring the MUST >=2026 MATLAB structure):
        param.xdcr    -- transducer geometry (XdcrParams)
        param.medium  -- acoustic medium properties (MediumParams)
        param.tx      -- transmit properties (TxParams)
        param.rx      -- receive properties (RxParams)

    For backward compatibility, every field of the old flat layout (e.g.
    param.fc, param.Nelements, param.RXdelay, ...) remains available as a
    deprecated property that reads/writes the corresponding nested field -
    see _DEPRECATED_ATTRS below. Existing code using the flat names keeps
    working unchanged; new code should prefer the nested form.
    """

    # flat (deprecated) name -> (sub-namespace, nested field name)
    _DEPRECATED_ATTRS = {
        'c': ('medium', 'c'),
        'attenuation': ('medium', 'attenuation'),
        'rho': ('medium', 'rho'),
        'beta': ('medium', 'beta'),

        'fc': ('xdcr', 'fc'),
        'f0': ('xdcr', 'fc'),  # alternative alias used in dasmtx/dasmtx3
        'pitch': ('xdcr', 'pitch'),
        'width': ('xdcr', 'width'),
        'kerf': ('xdcr', 'kerf'),
        'focus': ('xdcr', 'focus'),
        'height': ('xdcr', 'height'),
        'radius': ('xdcr', 'radius'),
        'bandwidth': ('xdcr', 'bandwidth'),
        'baffle': ('xdcr', 'baffle'),
        'Nelements': ('xdcr', 'nelements'),
        'elements': ('xdcr', 'elements'),

        'fe': ('tx', 'fe'),
        'TXnow': ('tx', 'now'),
        'TXfreqsweep': ('tx', 'freqsweep'),
        'TXapodization': ('tx', 'apodization'),
        'TXdelay': ('tx', 'delay'),
        'passive': ('tx', 'passive'),
        'PRF': ('tx', 'prf'),
        'PRP': ('tx', 'prp'),

        'fs': ('rx', 'fs'),
        'fnumber': ('rx', 'fnumber'),
        'RXdelay': ('rx', 'delay'),
        'RXangle': ('rx', 'angle'),
        't0': ('rx', 't0'),
    }

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        if not dict.__contains__(self, 'xdcr'):
            dict.__setitem__(self, 'xdcr', XdcrParams())
        if not dict.__contains__(self, 'medium'):
            dict.__setitem__(self, 'medium', MediumParams())
        if not dict.__contains__(self, 'tx'):
            dict.__setitem__(self, 'tx', TxParams())
        if not dict.__contains__(self, 'rx'):
            dict.__setitem__(self, 'rx', RxParams())

    def __setattr__(self, name, value):
        # Route through the property descriptor (if any) so that deprecated
        # flat names correctly land in the nested sub-namespaces, instead of
        # being blindly written as a dict entry (which dotdict would do).
        if isinstance(getattr(type(self), name, None), property):
            object.__setattr__(self, name, value)
        else:
            dict.__setitem__(self, name, value)

    def __contains__(self, key):
        if dict.__contains__(self, key):
            return True
        prop = getattr(type(self), key, None)
        if isinstance(prop, property):
            return getattr(self, key, None) is not None
        return False

    def get(self, key, default=None):
        prop = getattr(type(self), key, None)
        if isinstance(prop, property):
            value = getattr(self, key)
            return default if value is None else value
        return dict.get(self, key, default)

    @property
    def names(self):
        return {n.lower(): n for n in self._DEPRECATED_ATTRS}

    def check(self):
        """Validate all sub-structures and fill in their default values
        (mirrors MATLAB GETPARAM's CheckAndApplyDefaults, split per section)."""
        self.medium.check()
        self.xdcr.check()
        self.tx.check(self.xdcr)
        self.rx.check(self.xdcr)
        return self

    def getElementPositions(self):
        """
        Returns the position of each piezoelectrical element in the probe.
        """
        RadiusOfCurvature = self.radius
        NumberOfElements = self.Nelements

        if np.isinf(RadiusOfCurvature):
            #% Linear array
            xe =  (np.arange(NumberOfElements)-(NumberOfElements-1)/2)*self.pitch
            ze = np.zeros((1,NumberOfElements))
            THe = np.zeros_like(ze)
            h = np.zeros_like(ze)
        else:
            #% Convex array
            chord = 2*RadiusOfCurvature*np.sin(np.arcsin(self.pitch/2/RadiusOfCurvature)*(NumberOfElements-1))
            h = np.sqrt(RadiusOfCurvature**2-chord**2/4); #% apothem
            #% https://en.wikipedia.org/wiki/Circular_segment
            #% THe = angle of the normal to element #e with respect to the z-axis
            THe = np.linspace(np.arctan2(-chord/2,h),np.arctan2(chord/2,h),NumberOfElements)
            ze = RadiusOfCurvature*np.cos(THe)
            xe = RadiusOfCurvature*np.sin(THe)
            ze = ze-h
        return xe.reshape((1,-1)), ze.reshape((1,-1)), THe.reshape((1,-1)), h.reshape((1,-1))

    def getPulseSpectrumFunction(self, FreqSweep = None):
        if self.tx.now is None:
            self.TXnow = 1

        #-- FREQUENCY SPECTRUM of the transmitted pulse
        if FreqSweep is None:
            # We want a windowed sine of width PARAM.TXnow
            T = self.TXnow /self.fc
            wc = 2 * np.pi * self.fc
            pulseSpectrum = lambda w = None: 1j * (utils.mysinc(T * (w - wc) / 2) - utils.mysinc(T * (w + wc) / 2))
        else:
            # We want a linear chirp of width PARAM.TXnow
            # (https://en.wikipedia.org/wiki/Chirp_spectrum#Linear_chirp)
            T = self.TXnow / self.fc
            wc = 2 * np.pi * self.fc
            dw = 2 * np.pi * FreqSweep
            s2 = lambda w = None: np.multiply(np.sqrt(np.pi * T / dw) * np.exp(- 1j * (w - wc) ** 2 * T / 2 / dw),(utils.fresnelint((dw / 2 + w - wc) / np.sqrt(np.pi * dw / T)) + utils.fresnelint((dw / 2 - w + wc) / np.sqrt(np.pi * dw / T))))
            pulseSpectrum = lambda w = None: (1j * s2(w) - 1j * s2(- w)) / T
        return pulseSpectrum

    def getProbeFunction(self):
        #%-- FREQUENCY RESPONSE of the ensemble PZT + probe
        #% We want a generalized normal window (6dB-bandwidth = PARAM.bandwidth)
        #% (https://en.wikipedia.org/wiki/Window_function#Generalized_normal_window)
        #-- FREQUENCY RESPONSE of the ensemble PZT + probe
        # We want a generalized normal window (6dB-bandwidth = PARAM.bandwidth)
        # (https://en.wikipedia.org/wiki/Window_function#Generalized_normal_window)
        wc = 2 * np.pi * self.fc
        wB = self.bandwidth * wc / 100
        p = np.log(126) / np.log(2 * wc / wB)
        probeSpectrum_sqr = lambda w: np.exp(- np.power(np.abs(w - wc) / (wB / 2 / np.power(np.log(2), 1 / p)), p))
        # The frequency response is a pulse-echo (transmit + receive) response. A
        # square root is thus required when calculating the pressure field:
        probeSpectrum = lambda w: np.sqrt(probeSpectrum_sqr(w))
        return probeSpectrum


def _make_deprecated_param_alias(flat_name, subname, nested_name):
    """Builds a property that forwards param.<flat_name> to param.<subname>.<nested_name>."""
    def getter(self):
        warnings.warn(
            f'param.{flat_name} is deprecated; use param.{subname}.{nested_name} instead.',
            DeprecationWarning, stacklevel = 2)
        return getattr(getattr(self, subname), nested_name)

    def setter(self, value):
        warnings.warn(
            f'param.{flat_name} is deprecated; use param.{subname}.{nested_name} instead.',
            DeprecationWarning, stacklevel = 2)
        setattr(getattr(self, subname), nested_name, value)

    return property(getter, setter)

for _flat_name, (_subname, _nested_name) in Param._DEPRECATED_ATTRS.items():
    setattr(Param, _flat_name, _make_deprecated_param_alias(_flat_name, _subname, _nested_name))
del _flat_name, _subname, _nested_name


def getparam(probe: str) -> Param:
    #GETPARAM   Get parameters of a uniform linear or convex array
#   PARAM = GETPARAM opens a dialog box which allows you to select a
#   transducer whose parameters are returned in PARAM.

    #   PARAM = GETPARAM(PROBE), where PROBE is a string, returns the prameters
#   of the transducer given by PROBE.

    #   The structure PARAM is used in several functions of MUST (Matlab
#   UltraSound Toolbox). The structure returned by GETPARAM contains only
#   the fields that describe a transducer. Other fields may be required in
#   some MUST functions.

    #   PROBE can be one of the following:
#   ---------------------------------
#     1) 'L11-5v' (128-element, 7.6-MHz linear array)
#     2) 'L12-3v' (192-element, 7.5-MHz linear array)
#     3) 'C5-2v' (128-element, 3.6-MHz convex array)
#     4) 'P4-2v' (64-element, 2.7-MHz phased array)

    #   These are the <a
#   href="matlab:web('https://verasonics.com/verasonics-transducers/')">Verasonics' transducers</a>.
#   Feel free to complete this list for your own use.

    #   PARAM is a structure that contains the following fields:
#   --------------------------------------------------------
#   1) PARAM.Nelements: number of elements in the transducer array
#   2) PARAM.fc: center frequency (in Hz)
#   3) PARAM.pitch: element pitch (in m)
#   4) PARAM.width: element width (in m)
#   5) PARAM.kerf: kerf width (in m)
#   6) PARAM.bandwidth: 6-dB fractional bandwidth (in #)
#   7) PARAM.radius: radius of curvature (in m, Inf for a linear array)
#   8) PARAM.focus: elevation focus (in m)
#   9) PARAM.height: element height (in m)


    #   Example:
#   -------
#   #-- Generate a focused pressure field with a phased-array transducer
#   # Phased-array @ 2.7 MHz:
#   param = getparam('P4-2v');
#   # Focus position:
#   x0 = 2e-2; z0 = 5e-2;
#   # TX time delays:
#   dels = txdelay(x0,z0,param);
#   # Grid:
#   x = linspace(-4e-2,4e-2,200);
#   z = linspace(param.pitch,10e-2,200);
#   [x,z] = meshgrid(x,z);
#   y = zeros(size(x));
#   # RMS pressure field:
#   P = pfield(x,y,z,dels,param);
#   imagesc(x(1,:)*1e2,z(:,1)*1e2,20*log10(P/max(P(:))))
#   hold on, plot(x0*1e2,z0*1e2,'k*'), hold off
#   colormap hot, axis equal tight
#   caxis([-20 0])
#   c = colorbar;
#   c.YTickLabel{end} = '0 dB';
#   xlabel('[cm]')


    #   This function is part of <a
#   href="matlab:web('https://www.biomecardio.com/MUST')">MUST</a> (Matlab UltraSound Toolbox).
#   MUST (c) 2020 Damien Garcia, LGPL-3.0-or-later

    #   See also TXDELAY, PFIELD, SIMUS, GETPULSE.

    #   -- Damien Garcia -- 2015/03, last update: 2020/07
#   website: <a
#   href="matlab:web('https://www.biomecardio.com')">www.BiomeCardio.com</a>
    param = Param()
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
