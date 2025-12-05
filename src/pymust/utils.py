import numpy as np, scipy, scipy.interpolate, multiprocessing, multiprocessing.pool
from abc import ABC
import inspect, matplotlib, pickle, os, matplotlib.pyplot as plt, copy
from collections import deque
import numbers



class dotdict(dict, ABC):
    """Copied from https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary"""
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def ignoreCaseInFieldNames(self):
        """Convert all field names to lower case"""
        names = self.names
        todelete =[]
        for k, v in self.items():
            if k.lower() in names and k in names:
                if k.lower() == k:
                    continue
                elif names[k] in self:
                    raise ValueError(f'Repeated key {k}')
                else:
                    self[names[k]] = v
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
    @classmethod
    def getQuickOptions(cls):
        options = cls()
        options.dBThresh = -20
        options.ElementSplitting = 1
        options.FullFrequencyDirectivity = False
        options.FrequencyStep = 1.5
        return options

class Param(dotdict):
    @property 
    def names(self):
        """
        check the possible names 
        """
        names = {'attenuation','baffle','bandwidth','c','fc',
            'fnumber','focus','fs','height','kerf','movie','Nelements',
            'passive','pitch','radius','RXangle','RXdelay'
            'TXapodization','TXfreqsweep','TXnow','t0','width'}
        return {n.lower(): n for n in names}
    
    def checkTransducer(self):
        """
        Check whether the description of the transducer is correct.
        """
        #%---------------------------%
        #% Check the PARAM structure %
        #%---------------------------%

        param = self.ignoreCaseInFieldNames()

        #-- 0) Number of elements
        assert 'Nelements' in param, 'The number of elements (PARAM.Nelements) is required.'


        #-- 1) Center frequency (in Hz)
        assert 'fc' in param, 'A center frequency value (PARAM.fc) is required.'
        fc = param.fc # central frequency (Hz)
        assert isnumeric(fc) and np.isscalar(fc) and fc>0, 'The central frequency must be positive.'

        #%-- 2) Pitch (in m)
        assert'pitch' in param,'A pitch value (PARAM.pitch) is required.'
        pitch = param.pitch
        assert isnumeric(pitch) and np.isscalar(pitch) and pitch>0, 'The pitch must be positive.'

        #%-- 3) Element width and/or Kerf width (in m)
        if 'width' in param and 'kerf' in param:
            assert np.abs(pitch-param.width-param.kerf)< eps('single'), 'The pitch must be equal to (kerf width + element width).'
        elif 'kerf' in param:
            param.width = pitch-param.kerf
        elif 'width' in param:
            param.kerf = pitch-param.width
        else:
            raise ValueError('An element width (PARAM.width) or kerf width (PARAM.kerf) is required.')

        #%-- 4) Elevation focus (in m)
        if 'focus' not in param:
            param.focus = np.inf # default = no elevation focusing

        Rf = param.focus
        assert isnumeric(Rf) and np.isscalar(Rf) and Rf>0, 'The element focus must be positive.'

        #%-- 5) Element height (in m)
        if  'height' not in param:
            param.height = np.inf # default = line array

        ElementHeight = param.height
        assert isnumeric(ElementHeight) and np.isscalar(ElementHeight) and ElementHeight>0,'The element height must be positive.'

        #%-- 6) Radius of curvature (in m) - convex array
        if 'radius' not in param:
            param.radius = np.inf # default = linear array

        RadiusOfCurvature = param.radius
        assert isnumeric(RadiusOfCurvature) and np.isscalar(RadiusOfCurvature) and RadiusOfCurvature>0,'The radius of curvature must be positive.'

        #%-- 7) Fractional bandwidth at -6dB (in %)
        if 'bandwidth' not in param:
            param.bandwidth = 75

        assert param.bandwidth>0 and param.bandwidth<200, 'The fractional bandwidth at -6 dB (PARAM.bandwidth, in %) must be in ]0,200['

        #%-- 8) Baffle
        #   An obliquity factor will be used if the baffle is not rigid
        #%   (default = SOFT baffle)
        if  'baffle' not in param:
            param.baffle = 'soft' #  default

        if param.baffle in ['rigid', 'soft'] or np.isscalar(param.baffle):
            if isinstance(param.baffle, numbers.Number) and  param.baffle <= 0: 
                raise ValueError('The "baffle" field scalar must be positive')
        else:
            raise ValueError('The "baffle" field must be "rigid","soft" or a positive scalar')
        
    def checkTXParameters(self, delaysTX = None, apodTX = None):
        delaysTX = delaysTX.copy()
        #Check the transmit delays
        assert isnumeric(delaysTX) and all(delaysTX[~np.isnan(delaysTX)]>=0),  'DELAYS must be a nonnegative array.'

        NumberOfElements = delaysTX.shape[1]
        # Note: param.Nelements can be required in other functions of the
        #       Matlab Ultrasound Toolbox
        if 'Nelements' in self:
            assert self.Nelements==NumberOfElements, 'DELAYS must be of length PARAM.Nelements.'
        self.Nelements = NumberOfElements

    
        # Note: delaysTX can be a matrix. This option can be used for MLT
        # (multi-line transmit) for example. In this case, each row represents a
        # delay series. For example, for a 4-MLT sequence with a 64-element phased
        # array, delaysTX has 4 rows and 64 columns, i.e. size(delaysTX) = [4 64].

        #delaysTX  should be a row vector
        if len(delaysTX.shape) == 1:
            delaysTX = delaysTX.reshape((1, -1))
        delaysTX = delaysTX.astype(np.float32)

        #Check the transmit delays
        assert all(delaysTX[~np.isnan(delaysTX)]>=0),  'DELAYS must be a nonnegative array.'

        #%-- 11) Transmit apodization (no unit)
        if  apodTX is None:
            apodTX = np.ones((1,NumberOfElements), dtype = np.float32)
        else:
            apodTX = apodTX.copy()
            if isinstance(apodTX, np.ndarray) and len(apodTX.shape) == 1:
                apodTX = apodTX.reshape((1, -1))
            assert (len(apodTX.shape) == 2 and apodTX.shape[0] == 1) and isnumeric(apodTX), 'PARAM.TXapodization must be a vector'
            assert apodTX.shape[1]==NumberOfElements, 'PARAM.TXapodization must be of length = (number of elements)'

        #% apodization is 0 where TX delays are NaN:
        idx = np.isnan(delaysTX)
        apodTX[0, np.any(idx, axis = 0)]= 0
        delaysTX[idx] = 0

        # 12) TX pulse: Number of wavelengths
        if 'TXnow' not in self:
            self.TXnow = 1

        NoW = self.TXnow
        assert np.isscalar(NoW) and isnumeric(NoW) and NoW>0, 'PARAM.TXnow must be a positive scalar.'

        #%-- 13) TX pulse: Frequency sweep for a linear chirp
        if 'TXfreqsweep' not in self or np.isinf(NoW):
            self.TXfreqsweep = None

        FreqSweep = self.TXfreqsweep
        assert FreqSweep is None or (np.isscalar(FreqSweep) and isnumeric(FreqSweep) and FreqSweep>0), 'PARAM.TXfreqsweep must be empty (windowed sine) or a positive scalar (linear chirp).'
        return delaysTX, apodTX

    def obtainFrequencies(self, options, maxT = None):
        #%-- FREQUENCY STEP
        if options.isSIMUS or options.isMKMOVIE: #% PFIELD has been called by SIMUS or MKMOVIE
            df = options.FrequencyStep
        else: #% We are in PFIELD only (i.e. not called by SIMUS or MKMOVIE)
            #% The frequency step df is chosen to avoid interferences due to
            #% inadequate discretization.
            #% -- df = frequency step (must be sufficiently small):
            #% One has exp[-i(k r + w delay)] = exp[-2i pi(f r/c + f delay)] in the Eq.
            #% One wants: the phase increment 2pi(df r/c + df delay) be < 2pi.
            #% Therefore: df < 1/(r/c + delay).
            df = 1/maxT
            df = options.FrequencyStep*df
            #% note: df is here an upper bound; it will be recalculated below
            self.df = df

        #%-- FREQUENCY SAMPLES
        Nf = int(2*np.ceil(self.fc/df)+1) # number of frequency samples
        f = np.linspace(0,2*self.fc,Nf) # frequency samples
        df = f[1]  #% update the frequency step
        #%- we keep the significant components only by using options.dBThresh
        S = np.abs(self.pulseSpectrum(2*np.pi*f)*self.probeSpectrum(2*np.pi*f))

        GdB = 20*np.log10(1e-200 + S/np.max(S))# % gain in dB
        id = np.where(GdB >options.dBThresh)
        IDX = np.zeros(f.shape) != 0.
        IDX[id[0][0]:id[0][-1]+1] = True

        f = f[IDX]
        return f
    
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
        if 'TXnow' not in self:
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
