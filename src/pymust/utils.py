import numpy as np, scipy, scipy.interpolate, multiprocessing, multiprocessing.pool
from abc import ABC
import inspect, matplotlib, pickle, os, matplotlib.pyplot as plt, copy
from collections import deque


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

class Param(dotdict):
    @property 
    def names(self):
        names = {'attenuation','baffle','bandwidth','c','fc',
            'fnumber','focus','fs','height','kerf','movie','Nelements',
            'passive','pitch','radius','RXangle','RXdelay'
            'TXapodization','TXfreqsweep','TXnow','t0','width'}
        return {n.lower(): n for n in names}
    
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
def polarplot(x, z, v, cmap = 'gray',background = 'black', probeUpward = True):
    plt.pcolormesh(x, z, v, cmap = cmap, shading='gouraud')
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
