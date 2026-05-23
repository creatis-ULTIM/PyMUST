"""
.. note:: Documentation auto-generated with Claude (claude-sonnet-4-6).
"""
from . import utils, pfield, getpulse, numericalEngine
import logging, copy, multiprocessing, functools
import numpy as np

# pfield wrapper so it is compatible with multiprocessing. Needs to be defined in a global scope
def pfieldParallel(x: np.ndarray, y: np.ndarray, z: np.ndarray, RC: np.ndarray, delaysTX: np.ndarray, param: utils.Param, options: utils.Options):
    options = options.copy()
    options.ParPool = False # No parallel within the parallel
    options.RC = RC
    _, RFsp, idx =  pfield(x, y, z, delaysTX, param, options)
    return RFsp, idx


def simus(*varargin):
    """Simulate ultrasound RF signals for a linear or convex array.

    Simulates RF signals received by a uniform linear or convex array
    insonifying a medium of point scatterers.  Internally calls
    :func:`pfield` for each transmit/receive combination.

    Calling conventions::

        RF = simus(x, y, z, RC, delays, param)
        RF = simus(x, z, RC, delays, param)      # 2-D (y omitted or [])
        RF = simus(x, [], z, RC, delays, param)  # explicit 2-D
        RF = simus(..., options)                 # with advanced options

    Parameters
    ----------
    x, y, z : np.ndarray
        Scatterer coordinates (m). *y* may be omitted or empty for 2-D
        simulations (elevation focusing is ignored, computation is faster).
    RC : np.ndarray
        Reflection coefficients (same shape as *x*, *y*, *z*).
    delays : np.ndarray
        TX time delays (s); vector of length ``Nelements``, or matrix of
        shape ``(N_laws, Nelements)`` for MLT sequences.
    param : utils.Param
        Transducer and medium parameters:

        *Transducer*

        - ``fc``          : center frequency (Hz, required)
        - ``pitch``       : element pitch (m, required)
        - ``width``       : element width (m) **or** ``kerf`` (m, required)
        - ``focus``       : elevation focus depth (m, default ``np.inf``)
        - ``height``      : element height (m, default ``np.inf``)
        - ``radius``      : radius of curvature (m, default ``np.inf``)
        - ``bandwidth``   : 6-dB fractional bandwidth % (default 75)
        - ``baffle``      : ``'soft'`` (default), ``'rigid'``, or scalar α

        *Medium*

        - ``c``           : speed of sound (m/s, default 1540)
        - ``attenuation`` : attenuation (dB/cm/MHz, default 0)

        *Transmit*

        - ``TXapodization`` : TX apodization window (default: uniform)
        - ``TXnow``         : TX pulse length in wavelengths (default 1)
        - ``TXfreqsweep``   : frequency sweep for a linear chirp (default [])

        *Receive*

        - ``fs``      : sampling frequency (Hz, default 4·fc)
        - ``RXdelay`` : receive law delays (s, default 0)

    options : utils.Options, optional
        Advanced options:

        - ``dBThresh``               : amplitude threshold in dB (default -100)
        - ``FrequencyStep``          : frequency step scaling factor (default 1)
        - ``FullFrequencyDirectivity``: full frequency-dep. directivity (default False)
        - ``ElementSplitting``       : number of sub-elements per element
        - ``WaitBar``                : display progress bar (default True)
        - ``ParPool``                : enable parallel computation (default False)

    Returns
    -------
    RF : np.ndarray
        Simulated RF signals of shape ``(nsamples, Nelements)``.

    Examples
    --------
    >>> from pymust import getparam, txdelay, simus
    >>> import numpy as np
    >>> param = getparam('P4-2v')
    >>> param.fs = 4 * param.fc
    >>> dels = txdelay(0, 3e-2, param)
    >>> x = np.zeros(6); z = np.linspace(1e-2, 10e-2, 6)
    >>> RC = np.ones(6)
    >>> RF = simus(x, z, RC, dels, param)

    Notes
    -----
    Translated from the MATLAB MUST toolbox (Damien Garcia, 2017–2022).
    MUST (c) 2020 Damien Garcia, LGPL-3.0-or-later

    References
    ----------
    Shahriari S, Garcia D. Meshfree simulations of ultrasound vector flow
    imaging using smoothed particle hydrodynamics. Phys Med Biol,
    2018;63:205011.

    See Also
    --------
    pfield, txdelay, mkmovie, getparam, getpulse
    """

    returnTime = False #NoteGB: Set to True if you want to return the time, but quite a mess right now with the matlab style arguments

    nargin = len(varargin)
    if nargin<= 3 or nargin > 7:
        raise ValueError("Wrong number of input arguments.")
    #%-- Input variables: X,Y,Z,DELAYS,PARAM,OPTIONS
    x = varargin[0]

    if nargin ==5: # simus(X,Z,RC,DELAYS,PARAM)
            y = None
            z = varargin[1]
            RC = varargin[2]
            delaysTX = varargin[3]
            param = varargin[4]
            options = utils.Options()
    elif nargin == 6: # simus(X,Z,RC,DELAYS,PARAM,OPTIONS)
            if isinstance(varargin[4], utils.Param): #% simus(X,Z,RC,DELAYS,PARAM,OPTIONS)
                y = None
                z = varargin[1]
                RC = varargin[2]
                delaysTX = varargin[3]
                param = varargin[4]
                options = copy.deepcopy(varargin[5])
            else: # % simus(X,Y,Z,RC,DELAYS,PARAM)
                y = varargin[1]
                z = varargin[2]
                RC = varargin[3]
                delaysTX = varargin[4]
                param = varargin[5]
                options = utils.Options()
    else: # simus(X,Y,Z,RC,DELAYS,PARAM,OPTIONS)
                y = varargin[1]
                z = varargin[2]
                RC = varargin[3]
                delaysTX = varargin[4]
                param = varargin[5]
                options = copy.deepcopy(varargin[6])
    assert isinstance(param, utils.Param),'PARAM must be a structure.'

    #%-- Elevation focusing and X,Y,Z size
    if utils.isEmpty(y):
        ElevationFocusing = False
        assert x.shape == z.shape and x.shape == RC.shape, 'X, Z, and RC must be of same size.'
    else:
        ElevationFocusing = True
        assert x.shape == z.shape and x.shape == RC.shape and y.shape == x.shape,  'X, Y, Z, and RC must be of same size.'

    if len(x.shape) ==0:
         return np.array([]), np.array([])


    #%------------------------%
    #% CHECK THE INPUT SYNTAX % 
    #%------------------------%


    param = param.ignoreCaseInFieldNames()
    options = options.ignoreCaseInFieldNames()
    options.CallFun = 'simus'

    # GB TODO: wait bar + parallelisation
    #%-- Wait bar
    #if ~isfield(options,'WaitBar')
    #    options.WaitBar = true;
    #end
    #assert(isscalar(options.WaitBar) && islogical(options.WaitBar),...
    #    'OPTIONS.WaitBar must be a logical scalar (true or false).')

    #%-- Parallel pool
    #if ~isfield(options,'ParPool')
    #    options.ParPool = False
    #end

    #%-- Check if syntax errors may appear when using PFIELD
    #try:
    #    opt = options
    #    opt.ParPool = false;
    #    opt.WaitBar = false;
    #    [~,param] = pfield([],[],delaysTX,param,opt);
    #catch ME
    #    throw(ME)
    #end

    #%-- Sampling frequency (in Hz)
    if not utils.isfield(param,'fs'):
        param.fs = 4*param.fc; #% default

    assert param.fs>=4*param.fc,'PARAM.fs must be >= 4*PARAM.fc.'

    NumberOfElements = param.Nelements # % number of array elements

    #%-- Receive delays (in s)
    if not utils.isfield(param,'RXdelay'):
        param.RXdelay = np.zeros((1,NumberOfElements), dtype = np.float32)
    else:
        assert  isinstance(param.RXdelay, np.ndarray) and utils.isnumeric(param.RXdelay), 'PARAM.RXdelay must be a vector'
        assert param.RXdelay.shape[1] ==NumberOfElements, 'PARAM.RXdelay must be of length = (number of elements)'
        param.RXdelay = param.RXdelay.reshape((1,NumberOfElements))

    #%-- dB threshold (in dB: faster computation if lower value)
    if not utils.isfield(options,'dBThresh'):
        options.dBThresh = -100; # % default is -100dB in SIMUS

    assert np.isscalar(options.dBThresh) and utils.isnumeric(options.dBThresh) and options.dBThresh<0,'OPTIONS.dBThresh must be a negative scalar.'

    #%-- Frequency step (scaling factor)
    #% The frequency step is determined automatically. It is tuned to avoid
    #% aliasing in the temporal domain. The frequency step can be adjusted by
    #% using a scaling factor. For a smoother result, you may use a scaling
    #% factor<1.
    if not utils.isfield(options,'FrequencyStep'):
        options.FrequencyStep = 1

    assert np.isscalar(options.FrequencyStep) and utils.isnumeric(options.FrequencyStep) and  options.FrequencyStep>0, 'OPTIONS.FrequencyStep must be a positive scalar.'
    
    if options.FrequencyStep>1:
       logging.warning('MUST:FrequencyStep', 'OPTIONS.FrequencyStep is >1: aliasing may be present!')
    
    if not utils.isfield(param, 'c'):
         param.c = 1540 #default sound speed in soft tissue



    #%-------------------------------%
   # % end of CHECK THE INPUT SYNTAX %
   # %-------------------------------%
    
    #GB NOTE: same as in pfield, put in param ?
    #%-- Centers of the tranducer elements (x- and z-coordinates)
    xe, ze, THe, h= param.getElementPositions()

    #%-- Maximum distance
    d2 = (x.reshape((-1,1))-xe)**2+(z.reshape((-1,1))-ze)**2
    maxD = np.sqrt(np.max(d2)) #% maximum element-scatterer distance
    _, tp = getpulse.getpulse(param, 2)
    maxD = maxD + tp[-1] * param.c #add pulse length

    #%-- FREQUENCY SAMPLES
    valid_tx_delays = np.array([e for e in delaysTX.flatten() if not np.isnan(e)])
    df = 1/2/(2*maxD/param.c + np.max(np.concatenate((valid_tx_delays,param.RXdelay.flatten())))) # % to avoid aliasing in the time domain
    # df = 1/2/(2*maxD/param.c + np.max(delaysTX.flatten() + param.RXdelay.flatten())) # % to avoid aliasing in the time domain
    df = df*options.FrequencyStep
    Nf = 2*int(np.ceil(param.fc/df))+1 # % number of frequency samples
    #%-- Run PFIELD to calculate the RF spectra
    RFspectrum = np.zeros((Nf,NumberOfElements), dtype = np.complex64)# % will contain the RF spectra
    options.FrequencyStep = df

    #%- run PFIELD in a parallel pool (NW workers)
    if options.get('ParPool', False):
        if 'numericalEngine' in options and not options['numericalEngine'].isNumpy:        
             raise NotImplemented("Cannot use a numerical engine other than numpy for parallel computing")
        with options.getParallelPool() as pool:
            idx = options.getParallelSplitIndices(x.shape[1])

            RS = pool.starmap(functools.partial(pfieldParallel, delaysTX = delaysTX, param = param, options = options),
                            [ ( x[:,i:j],
                                y[:,i:j] if not utils.isEmpty(y) else None, 
                                z[:,i:j], 
                                RC[:,i:j]) for i,j in idx ])
            

            for (RFsp, idx_spectrum) in RS: 
                RFspectrum[idx_spectrum, :] += RFsp

    #    end
    else:
        #%- no parallel pool 
        options.RC =  RC
        # 
        extra_args = {}
        if 'engine' in options:
             extra_args['numericalEngine'] = options['numericalEngine']
        _, RFsp,idx = pfield(x,y,z,delaysTX,param,options, **extra_args)

        RFspectrum[idx,:]  = RFsp

    #%-- RF signals (in the time domain)
    nf = int(np.ceil(param.fs/2/param.fc*(Nf-1)))
    RF = np.fft.irfft(np.conj(RFspectrum),nf, axis = 0)
    RF = RF[:(nf + 1)//2] #*param.fs/4/param.fc

    #%-- Zeroing the very small values
    RelThresh = 1e-5#; % -100 dB
    tmp2= lambda RelRF: 0.5*(1+np.tanh((RelRF-RelThresh)/(RelThresh/10)))
    tmp = lambda RelRF: np.round(tmp2(RelRF)/(RelThresh/10))*(RelThresh/10)
    RF = RF*tmp(np.abs(RF)/np.max(np.abs(RF)))
    if returnTime: 
        return RF,RFspectrum, np.arange(RF.shape[0])/param.fs
    else:
         return RF,RFspectrum

