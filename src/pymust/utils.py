import numpy as np, scipy, scipy.interpolate, multiprocessing, multiprocessing.pool
from abc import ABC
import inspect, matplotlib, pickle, os, matplotlib.pyplot as plt, copy, warnings
from collections import deque


class dotdict(dict, ABC):
    """Copied from https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary"""
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def ignoreCaseInFieldNames(self):
        """Normalize field names to their canonical casing (case-insensitive aliasing).

        Uses setattr/getattr (not raw dict access) so that on subclasses where a
        canonical name is backed by a property (e.g. Param's deprecated flat
        aliases), the value is correctly routed through that property instead of
        being dropped into an orphaned dict entry.
        """
        names = self.names
        todelete = []
        for k in list(self.keys()):
            canonical = names.get(k.lower())
            if canonical is None or k == canonical:
                continue
            if canonical in self:
                raise ValueError(f'Repeated key {k}')
            setattr(self, canonical, self[k])
            todelete.append(k)
        for k in todelete:
            del self[k]
        return self
    def copy(self):
        return copy.deepcopy(self)
    def __getstate__(self):
        d = {k : v for k,v in self.items()}
        return d
    def __setstate__(self, d):
        for k, v in self.items():
            self[k] = v
        
class Options(dotdict):
    default_Number_Workers = multiprocessing.cpu_count()
    @property 
    def names(self):
        names = {'dBThresh','ElementSplitting',
                'FullFrequencyDirectivity','FrequencyStep','ParPool',
                'WaitBar'}
        return {n.lower(): n for n in names}
    
    def setParPool(self, workers, mode = 'process'):
        if mode not in ['process', 'thread']:
            raise ValueError('ParPoolMode must be either "process" or "thread"')
        self.ParPool_NumWorkers = workers
        self.ParPoolMode = mode
    
    def getParallelPool(self):
        workers = self.get('ParPool_NumWorkers', self.default_Number_Workers)
        mode = self.get('ParPoolMode', 'thread')
        if mode == 'process':
            pool = multiprocessing.Pool(workers)
        elif mode == 'thread':
            pool = multiprocessing.pool.ThreadPool(workers)
        else:
             raise ValueError('ParPoolMode must be either "process" or "thread"')
        return pool
    
    def getParallelSplitIndices(self, N,n_threads = None):
        if hasattr(N, '__len__'):
            N = len(N)
        assert isinstance(N, int), 'N must be an integer'

        n_threads = self.get('ParPool_NumWorkers', self.default_Number_Workers) if n_threads is None else n_threads
        #Create indices for parallel processing, split in workers
        idx = np.arange(0, N, N//n_threads)

        #Repeat along new axis
        idx = np.stack([idx, np.roll(idx, -1)], axis = 1)
        idx[-1, 1] = N
        return idx

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
        assert isnumeric(self.c) and np.isscalar(self.c) and self.c > 0, \
            'PARAM.medium.c must be a positive scalar.'

        if self.attenuation is None:
            self.attenuation = 0
        assert isnumeric(self.attenuation) and np.isscalar(self.attenuation) and self.attenuation >= 0, \
            'PARAM.medium.attenuation must be a nonnegative scalar.'

        if self.rho is None:
            self.rho = 1050
        assert isnumeric(self.rho) and np.isscalar(self.rho) and self.rho > 0, \
            'PARAM.medium.rho must be a positive scalar.'

        if self.beta is None:
            self.beta = 4.5
        assert isnumeric(self.beta) and np.isscalar(self.beta) and self.beta >= 0, \
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

    def check(self):
        if self.fc is not None:
            assert isnumeric(self.fc) and np.isscalar(self.fc) and self.fc > 0, \
                'The center frequency (PARAM.xdcr.fc) must be positive.'

        if self.nelements is not None:
            assert isnumeric(self.nelements) and np.isscalar(self.nelements) and self.nelements > 0 \
                and self.nelements == round(self.nelements), \
                'The number of elements (PARAM.xdcr.nelements) is invalid.'

        if self.bandwidth is None:
            self.bandwidth = 75
        assert isnumeric(self.bandwidth) and np.isscalar(self.bandwidth) and 0 < self.bandwidth < 200, \
            'The fractional bandwidth (PARAM.xdcr.bandwidth) must be in ]0,200[.'

        if self.focus is None:
            self.focus = np.inf
        assert isnumeric(self.focus) and np.isscalar(self.focus) and self.focus > 0, \
            'The elevation focus (PARAM.xdcr.focus) must be positive.'

        if self.height is None:
            self.height = np.inf
        assert isnumeric(self.height) and np.isscalar(self.height) and self.height > 0, \
            'The element height (PARAM.xdcr.height) must be positive.'

        if self.radius is None:
            self.radius = np.inf
        assert isnumeric(self.radius) and np.isscalar(self.radius) and self.radius > 0, \
            'The radius of curvature (PARAM.xdcr.radius) must be positive.'

        if self.baffle is None:
            self.baffle = 'soft'
        if isinstance(self.baffle, str):
            assert self.baffle.lower() in ('rigid', 'soft'), \
                "The baffle (PARAM.xdcr.baffle) must be 'rigid' or 'soft'."
        else:
            assert isnumeric(self.baffle) and np.isscalar(self.baffle) and self.baffle > 0, \
                'The baffle scalar (PARAM.xdcr.baffle) must be positive.'

        # Pitch, width, and kerf must be mutually consistent (pitch = width + kerf)
        if self.pitch is not None:
            assert isnumeric(self.pitch) and np.isscalar(self.pitch) and self.pitch > 0, \
                'The pitch (PARAM.xdcr.pitch) must be positive.'
            if self.width is not None and self.kerf is not None:
                tol = 10 * eps() * max(abs(self.pitch), abs(self.width), abs(self.kerf), 1.0)
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
            assert isnumeric(self.kerf) and np.isscalar(self.kerf) and self.kerf >= 0, \
                'The kerf width (PARAM.xdcr.kerf) must be nonnegative.'
        if self.width is not None:
            assert isnumeric(self.width) and np.isscalar(self.width) and self.width > 0, \
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
            assert isnumeric(self.fe) and np.isscalar(self.fe) and self.fe > 0, \
                'The excitation frequency (PARAM.tx.fe) must be positive.'
            if xdcr.fc is not None:
                assert abs(self.fe - xdcr.fc) < (xdcr.fc * xdcr.bandwidth / 200), \
                    "The excitation frequency (PARAM.tx.fe) is outside the transducer's bandwidth."

        if self.now is None:
            self.now = 1
        assert np.isscalar(self.now) and isnumeric(self.now) and self.now > 0, \
            'PARAM.tx.now must be a positive scalar.'

        if self.freqsweep is not None:
            assert np.isscalar(self.freqsweep) and isnumeric(self.freqsweep) and self.freqsweep > 0, \
                'PARAM.tx.freqsweep must be None (windowed sine) or a positive scalar (linear chirp).'

        if xdcr.nelements is not None and self.apodization is None:
            self.apodization = np.ones(xdcr.nelements)
        if self.apodization is not None:
            apod = np.asarray(self.apodization)
            assert apod.ndim == 1 and isnumeric(apod), 'PARAM.tx.apodization must be a numeric vector.'
            if xdcr.nelements is not None:
                assert apod.size == xdcr.nelements, \
                    'PARAM.tx.apodization must have length = (number of elements).'

        if self.delay is not None:
            delay = np.asarray(self.delay)
            assert delay.ndim <= 2 and isnumeric(delay), \
                'PARAM.tx.delay must be a numeric vector or matrix.'
            if xdcr.nelements is not None:
                assert delay.shape[-1] == xdcr.nelements, \
                    'PARAM.tx.delay must be a row vector or a matrix whose columns correspond to elements.'

        if self.passive is None:
            self.passive = False
        assert isinstance(self.passive, (bool, np.bool_)) or (isnumeric(self.passive) and self.passive in (0, 1)), \
            'PARAM.tx.passive must be True, False, 0, or 1.'
        self.passive = bool(self.passive)

        if self.prf is not None:
            assert isnumeric(self.prf) and np.isscalar(self.prf) and np.isfinite(self.prf) and self.prf > 0, \
                'PARAM.tx.prf must be a positive scalar.'
        if self.prp is not None:
            assert isnumeric(self.prp) and np.isscalar(self.prp) and np.isfinite(self.prp) and self.prp > 0, \
                'PARAM.tx.prp must be a positive scalar.'
        if self.prf is not None and self.prp is not None:
            tol = 10 * eps() * max(abs(self.prf), abs(1 / self.prp), 1.0)
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
            assert isnumeric(self.fs) and np.isscalar(self.fs) and self.fs > 0, \
                'PARAM.rx.fs must be a positive scalar.'

        if xdcr.nelements is not None and self.delay is None:
            self.delay = np.zeros(xdcr.nelements)
        if self.delay is not None:
            delay = np.asarray(self.delay)
            assert delay.ndim == 1 and isnumeric(delay), 'PARAM.rx.delay must be a numeric vector.'
            if xdcr.nelements is not None:
                assert delay.size == xdcr.nelements, \
                    'PARAM.rx.delay must have length = (number of elements).'

        if self.angle is None:
            self.angle = 0
        assert isnumeric(self.angle) and np.isscalar(self.angle), 'PARAM.rx.angle must be a numeric scalar.'

        if self.fnumber is None:
            self.fnumber = 0
        else:
            assert isnumeric(self.fnumber) and np.isscalar(self.fnumber) and self.fnumber >= 0, \
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
        assert isnumeric(self.t0) and np.isscalar(self.t0) and self.t0 >= 0, \
            'PARAM.rx.t0 must be a nonnegative scalar.'
        return self


class Param(dotdict):
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
            pulseSpectrum = lambda w = None: 1j * (mysinc(T * (w - wc) / 2) - mysinc(T * (w + wc) / 2))
        else:
            # We want a linear chirp of width PARAM.TXnow
            # (https://en.wikipedia.org/wiki/Chirp_spectrum#Linear_chirp)
            T = self.TXnow / self.fc
            wc = 2 * np.pi * self.fc
            dw = 2 * np.pi * FreqSweep
            s2 = lambda w = None: np.multiply(np.sqrt(np.pi * T / dw) * np.exp(- 1j * (w - wc) ** 2 * T / 2 / dw),(fresnelint((dw / 2 + w - wc) / np.sqrt(np.pi * dw / T)) + fresnelint((dw / 2 - w + wc) / np.sqrt(np.pi * dw / T))))
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

# To maintain same notation as matlab
def interp1(y, xNew, kind):
    if kind == 'spline':
        kind = 'cubic' #3rd order spline
    interpolator = scipy.interpolate.interp1d(np.arange(len(y)), y, kind = kind) 
    return interpolator(xNew)    

def isnumeric(x):
    return isinstance(x, np.ndarray) or isinstance(x, int) or isinstance(x, float) or isinstance(x, np.number)

def iscomplex(x):
    return (isinstance(x, np.ndarray) and np.iscomplexobj(x)) or isinstance(x, complex)

def islogical(v):
    return isinstance(v, bool)

def isfield(d, k ):
    return k in d

mysinc = lambda x = None: np.sinc(x / np.pi) # [note: In MATLAB/numpy, sinc is sin(pi*x)/(pi*x)]


def shiftdim(array, n=None):
    """
    From stack overflow https://stackoverflow.com/questions/67584148/python-equivalent-of-matlab-shiftdim
    """
    if n is not None:
        if n >= 0:
            axes = tuple(range(len(array.shape)))
            new_axes = deque(axes)
            new_axes.rotate(n)
            return np.moveaxis(array, axes, tuple(new_axes))
        return np.expand_dims(array, axis=tuple(range(-n)))
    else:
        idx = 0
        for dim in array.shape:
            if dim == 1:
                idx += 1
            else:
                break
        axes = tuple(range(idx))
        # Note that this returns a tuple of 2 results
        return np.squeeze(array, axis=axes), len(axes)

def isEmpty(x):
    return  x is None or (isinstance(x, list) and len(x) == 0) or (isinstance(x, np.ndarray) and len(x) == 0)

def emptyArrayIfNone(x):
    if isEmpty(x):
        x =  np.array([])
    return x

def eps(s = 'single'):
    if s == 'single':
        return 1.1921e-07 
    else:
        raise ValueError()

def nextpow2(n):
    i = 1
    while (1 << i) < n:
        i += 1
    return i

def fresnelint(x): 
    # FRESNELINT Fresnel integral.
    
    # J = FRESNELINT(X) returns the Fresnel integral J = C + 1i*S.
    
    # We use the approximation introduced by Mielenz in
#       Klaus D. Mielenz, Computation of Fresnel Integrals. II
#       J. Res. Natl. Inst. Stand. Technol. 105, 589 (2000), pp 589-590
    
    siz0 = x.shape
    x = x.flatten()

    issmall = np.abs(x) <= 1.6
    c = np.zeros(x.shape)
    s = np.zeros(x.shape)
    # When |x| < 1.6, a Taylor series is used (see Mielenz's paper)
    if np.any(issmall):
        n = np.arange(0,11)
        cn = np.concatenate([[1], np.cumprod(- np.pi ** 2 * (4 * n + 1) / (4 * (2 * n + 1) *(2 * n + 2)*(4 * n + 5)))])
        sn = np.concatenate([[1],np.cumprod(- np.pi ** 2 * (4 * n + 3) / (4 * (2 * n + 2)*(2 * n + 3)*(4 * n + 7)))]) * np.pi / 6
        n = np.concatenate([n,[11]]).reshape((1,-1))
        c[issmall] = np.sum(cn.reshape((1,-1))*x[issmall].reshape((-1, 1))  ** (4 * n + 1), 1)
        s[issmall] = np.sum(sn.reshape((1,-1))*x[issmall].reshape((-1, 1)) ** (4 * n + 3), 1)
    
    # When |x| > 1.6, we use the following:
    if not np.all(issmall ):
        n = np.arange(0,11+1)
        fn = np.array([0.318309844,9.34626e-08,- 0.09676631,0.000606222,0.325539361,0.325206461,- 7.450551455,32.20380908,- 78.8035274,118.5343352,- 102.4339798,39.06207702])
        fn = fn.reshape((1, fn.shape[0]))
        gn = np.array([0,0.101321519,- 4.07292e-05,- 0.152068115,- 0.046292605,1.622793598,- 5.199186089,7.477942354,- 0.695291507,- 15.10996796,22.28401942,- 10.89968491])
        gn = gn.reshape((1, gn.shape[0]))

        fx = np.sum(np.multiply(fn,x[not issmall ] ** (- 2 * n - 1)), 1)
        gx = np.sum(np.multiply(gn,x[not issmall ] ** (- 2 * n - 1)), 1)
        c[not issmall ] = 0.5 * np.sign(x[not issmall ]) + np.multiply(fx,np.sin(np.pi / 2 * x[not issmall ] ** 2)) - np.multiply(gx,np.cos(np.pi / 2 * x[not issmall ] ** 2))
        s[not issmall ] = 0.5 * np.sign(x[not issmall ]) - np.multiply(fx,np.cos(np.pi / 2 * x[not issmall ] ** 2)) - np.multiply(gx,np.sin(np.pi / 2 * x[not issmall ] ** 2))
    
    f = np.reshape(c, siz0) + 1j * np.reshape(s, siz0)
    return f


# Plotting
def polarplot(x, z, v, cmap = 'gray',background = 'black', probeUpward = True, **kwargs):
    plt.pcolormesh(x, z, v, cmap = cmap, shading='gouraud', **kwargs)
    plt.axis('equal')
    ax = plt.gca()
    ax.set_facecolor(background)
    if probeUpward:
        ax.invert_yaxis()


def getDopplerColorMap():
    source_file_path = inspect.getfile(inspect.currentframe())
    with open( os.path.join(os.path.dirname(source_file_path), 'Data', 'colorMap.pkl'), 'rb') as f:
        dMap = pickle.load(f)
    new_cmap = matplotlib.colors.LinearSegmentedColormap('doppler', dMap)
    dopplerCM = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(),cmap=new_cmap)
    return dopplerCM

def applyDasMTX(M, IQ, imageShape):
    return (M @ IQ.flatten(order = 'F')).reshape(imageShape, order = 'F')
