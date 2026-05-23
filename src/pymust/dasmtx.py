"""
.. note:: Documentation auto-generated with Claude (claude-sonnet-4-6).
"""
import numpy as np
import scipy.optimize, itertools

import scipy, scipy.interpolate
from . import  utils


def dasmtx(SIG: np.ndarray, x: np.ndarray, z: np.ndarray, *varargin) -> scipy.sparse.spmatrix:
    """Build a delay-and-sum sparse matrix for 2-D beamforming.

    Returns the ``numel(x)``-by-``numel(SIG)`` sparse DAS matrix *M* that
    beamforms *SIG* at the grid points (*x*, *z*).  Once computed, *M* can be
    reused for many frames:

    .. code-block:: python

        M = dasmtx(SIG.shape, x, z, delays, param)
        bfSIG = (M @ SIG.flatten(order='F')).reshape(x.shape, order='F')

    To build a complex DAS matrix for I/Q data pass a purely imaginary size
    vector: ``dasmtx(1j*np.array(SIG.shape), x, z, delays, param)``.

    Parameters
    ----------
    SIG : np.ndarray or array-like
        RF or I/Q signal array of shape ``(nsamples, nelements)`` **or** a
        2-/3-element size vector (only the shape is used to build *M*).
    x : np.ndarray
        x-coordinates of beamforming points (m).
    z : np.ndarray
        z-coordinates of beamforming points (m); must have the same shape as *x*.
    *varargin :
        One of the following calling conventions (plus an optional trailing
        interpolation method string):

        - ``dasmtx(SIG, x, z, param)`` — uses ``param.TXdelay`` as delays.
        - ``dasmtx(SIG, x, z, delays, param)`` — explicit delay vector (s).

    param : utils.Param
        Parameter structure with the following fields:

        - ``fs``        : sampling frequency (Hz, required)
        - ``pitch``     : element pitch (m, required)
        - ``fc``        : center frequency (Hz, required for I/Q data)
        - ``c``         : speed of sound (m/s, default 1540)
        - ``t0``        : acquisition start time (s, default 0)
        - ``radius``    : radius of curvature (m, default ``np.inf``)
        - ``TXdelay``   : TX delays (s, required if not passed as *delays*)
        - ``fnumber``   : receive f-number (default 0 = full aperture;
          ``None`` = automatic from element directivity)
        - ``bandwidth`` : 6-dB fractional bandwidth % (default 75,
          used only when ``fnumber=None``)
        - ``width``     : element width (m, needed when ``fnumber=None``)
        - ``RXangle``   : receive angle for vector Doppler (rad, default 0)
        - ``passive``   : set ``True`` for passive imaging (default ``False``)
        - ``elements``  : element x-(and z-) positions (m), overrides pitch

    method : str, optional
        Interpolation method (case-insensitive, default ``'linear'``):
        ``'nearest'``, ``'linear'``, ``'quadratic'``, ``'lanczos3'``,
        ``'5points'``, ``'lanczos5'``.

    Returns
    -------
    M : scipy.sparse.coo_matrix
        Sparse DAS matrix of shape ``(numel(x), nsamples*nelements)``.

    Notes
    -----
    Translated from the MATLAB MUST toolbox (Damien Garcia, 2017–2022;
    Gabriel Bernardino, 2023). MUST (c) 2020 Damien Garcia, LGPL-3.0-or-later

    References
    ----------
    Perrot V, Polichetti M, Varray F, Garcia D. So you think you can DAS?
    A viewpoint on delay-and-sum beamforming. Ultrasonics 111, 106309, 2021.

    See Also
    --------
    simus, txdelay, rf2iq, getparam
    """



    #%------------------------%
    #% CHECK THE INPUT SYNTAX %
    #%------------------------%

    NArg = len(varargin) 
    assert NArg>0,'Not enough input arguments.'
    assert NArg<4,'Too many input arguments.'
    #% assert(nargout<3,'Too many output arguments.')

    assert x.shape == z.shape,'X and Z must of same size.'
    if  np.prod(SIG.shape) in [2, 3]:
        SIG = SIG.flatten()
        nl = int(abs(SIG[0]))
        nc = int(abs(SIG[1]))
    else:
        nl,nc = SIG.shape[0], SIG.shape[1]


    #% check if we have I/Q signals
    isIQ = utils.iscomplex(SIG)

    #%-- Check input parameters
    if isinstance(varargin[-1], str):
        method = varargin[NArg - 1]
        NArg = NArg-1
    else:
        method = 'linear'
        NArg = NArg


    if NArg==1: #% DASMTX(SIG,x,z,param)
        if isinstance(varargin[0], utils.Param):
            param = varargin[0]
            param.ignoreCaseInFieldNames()
        else:
            raise ValueError('The structure PARAM is required.')

        assert utils.isfield(param,'TXdelay'), 'A TX delay vector (PARAM.TXdelay or DELAYS) is required.'
        delaysTX = param.TXdelay

    else: #% NArg=5: #DASMTX(SIG,x,z,delaysTX,param)
        delaysTX = varargin[0]
        param = varargin[1]
        assert isinstance(param, utils.Param), 'The structure PARAM is required.'
        param.ignoreCaseInFieldNames()

        if utils.isfield(param,'TXdelay'): #% DASMTX(SIG,x,z,delaysTX,param)
            assert np.allclose(delaysTX.flatten(), param.TXdelay.flatten()),'If both specified, PARAM.TXdelay and DELAYS must be equal.'



    #%-- Interpolation method
    if method.lower() not in ['nearest','linear','quadratic','lanczos3','5points','lanczos5']:
        raise ValueError('METHOD must be "nearest", "linear", "quadratic", "Lanczos3", "5points" or "Lanczos5".')


    #%-- Propagation velocity (in m/s)
    if not utils. isfield(param,'c'):
        param.c = 1540; #% longitudinal velocity in m/s


    #%-- Sampling frequency (in Hz)
    if param.get('fs', None) is None:
        raise ValueError('A sampling frequency (PARAM.fs) is required.')

    #%-- f-number
    if not utils.isfield(param,'fnumber'):
        param.fnumber = 0; #% f-number (default = full aperture)
    elif param.get('fnumber', 'None') is None:
        # do nothing:
        # The f-number will be determined automatically
        pass
    else:
        assert np.isscalar(param.fnumber) and utils.isnumeric(param.fnumber), 'If not empty, PARAM.fnumber must be a scalar.'
        assert param.fnumber>=0, 'PARAM.fnumber must be non-negative.'


    #%-- Acquisition start time (in s)
    if not utils.isfield(param,'t0'):
        param.t0 = np.zeros((1,1)) #% acquisition start time in s


    #%-- Pitch & width or kerf (in m)
    if not utils.isfield(param,'pitch'):
        raise ValueError('A pitch value (PARAM.pitch) is required.')

    if param.get('fnumber', None) is None:
        #% NOTE:
        #% An element width or a kerf width is required if the f-number is []
        if utils.isfield(param,'width') and utils.isfield(param,'kerf'):
            assert abs(param.pitch-param.width-param.kerf)<utils.eps(), 'The pitch must be equal to (kerf width + element width).'
        elif utils.isfield(param,'kerf'):
            param.width = param.pitch-param.kerf
        elif utils.isfield(param,'width'):
            param.kerf = param.pitch-param.width
        else:
            raise ValueError('An element width (PARAM.width) or kerf width (PARAM.kerf) is required if PARAM.fnumber = [].')

        ElementWidth = param.width


    #%-- -6dB bandwidth (in %)
    if not utils.isfield(param,'bandwidth'):
        param.bandwidth = 75

    if param.get('fnumber', None) is None:
        assert param.bandwidth>0 and param.bandwidth<200, 'The fractional bandwidth at -6 dB (PARAM.bandwidth, in %) must be in ]0,200['


    #%-- Radius of curvature (in m)
    #% for a convex array
    if not utils.isfield(param,'radius'):
        param.radius = np.inf #% default = linear array

    RadiusOfCurvature = param.radius
    isLINEAR = np.isinf(RadiusOfCurvature)

    #%-- Reception angle (in rad) -- [Advanced option for vector Doppler] --
    if not utils.isfield(param,'RXangle') or param.RXangle==0:
        isRXangle = False
        param.RXangle = 0
    else:
        #%- this option is not available for convex arrays
        assert np.isinf(RadiusOfCurvature), 'PARAM.RXangle must be 0 with a convex array.'
        #%-
        isRXangle = True
        cosRX = np.cos(param.RXangle)
        tanRX = np.tan(param.RXangle)
        if np.isscalar(param.RXangle):
            cosRX = cosRX*np.ones(x.shape)
            tanRX = tanRX*np.ones(x.shape)


    # NoteGB: Temporary, while checking the code of virtual sources in the DAS matrix
    if not utils.isfield(param,'useVirtualSource'):
        useVirtualSource = False
    else:
        useVirtualSource = param.useVirtualSource


    #%-- Passive imaging
    if not utils.isfield(param,'passive'):
        param.passive = False
    else:
        assert utils.islogical(param.passive), 'PARAM.passive must be a boolean (false or true)'


    #%-- Number of elements
    assert isinstance(delaysTX,np.ndarray), 'delaysTX must be a numpy array.'
    if len(delaysTX.shape) == 1:
        delaysTX = delaysTX.reshape((1, -1))
    if delaysTX.shape[0] == nc and delaysTX.shape[1] != nc:
        delaysTX = delaysTX.T

    assert delaysTX.shape[1]==nc, 'DELAYS and/or PARAM.TXdelay must be vectors of length size(SIG,2).'
    #% Note: param.Nelements can be required in other functions of the
    #%       Matlab Ultrasound Toolbox
    if utils.isfield(param,'Nelements'):
        assert param.Nelements==nc, 'PARAM.TXdelay or DELAYS must be of length PARAM.Nelements.'


    #%-- Locations of the transducer elements (!if PARAM.elements is given!)
    if utils.isfield(param,'elements'):
        assert len(param.elements.shape) == 2 and (1 in param.elements.shape or 2 in param.elements.shape), \
        'PARAM.elements must be a vector, a 2-row or a 2-column matrix that contains the x- (and optionally z-)locations of the transducer elements.'
        if param.elements.shape[0] ==2:
            #% param.elements is a two-row matrix
            xe = param.elements[0, :]
            ze = param.elements[1, :]
        elif param.elements.shape[1 ]==2:
            #% param.elements is a two-column matrix
            xe = param.elements[:, 0]
            ze = param.elements[:, 1]
        else:
            #% param.elements is a vector
            xe = param.elements.reshape((1,-1))
            ze = np.zeros_like(xe)

        isPARAMelements = True
    else:
        isPARAMelements = False


    #%-- Center frequency (in Hz)
    if isIQ:
        if utils.isfield(param,'fc'):
            if utils.isfield(param,'f0'):
                assert abs(param.fc-param.f0)<utils.eps(), \
                    'A conflict exists for the center frequency: PARAM.fc and PARAM.f0 are different!'

        elif utils.isfield(param,'f0'):
            param.fc = param.f0 #% Note: param.f0 can also be used
        else:
            raise ValueError('A center frequency (PARAM.fc) is required with I/Q data.')
        
        wc = 2*np.pi*param.fc



    #%-------------------------------%
    #% end of CHECK THE INPUT SYNTAX %
    #%-------------------------------%



    #%-- Centers of the tranducer elements (x- and z-coordinates)
    xe, ze, THe, h = param.getElementPositions()

    #% note: THe = angle of the normal to element #e with respect to the z-axis


    #% some parameters
    fs = param.fs #% sampling frequency
    c = param.c   #% propagation velocity


    #% Interpolations of the TX delays for a better estimation of dTX
    idx = np.logical_not(np.isnan(delaysTX))
    assert np.sum(np.abs(np.diff(idx)))<3, 'Several simultaneous sub-apertures are not allowed.'

    if not param.passive:
        nTX = np.count_nonzero(idx)#; % number of transmitting elements
        if nTX>1:
            idxi = np.linspace(0,nTX -1,4*nTX)
            xTi = utils.interp1(xe[idx], idxi, kind = 'cubic')
            if isLINEAR:
                zTi = np.zeros_like(xTi)
            else:
                zTi = utils.interp1(ze[idx], idxi,'spline')
            delaysTXi = utils.interp1(delaysTX[idx],idxi,'spline')
        else:
            xTi = xe[idx]
            if isLINEAR:
                zTi =np.zeros_like(idx)
            else: 
                zTi = ze[idx]
            delaysTXi = delaysTX[idx]


    #%-- f-number (determined automatically if not given)
    #% The f-number is determined from the element directivity
    #% See the paper "So you think you can DAS?"
    if param.get('fnumber', None) is None:
        lambdaMIN = c/(param.fc*(1+param.bandwidth/200))
        RXa = abs(param.RXangle)
        #% Note: in Matlab, sinc(x) = sin(pi*x)/(pi*x)
        f = lambda th,width= ElementWidth,l= lambdaMIN: np.abs(np.cos(th+RXa)*np.sinc(width/l*np.sin(th+RXa))-0.71)

        xx = scipy.optimize.fminbound(f,0,np.pi/2-RXa,xtol= np.pi/100)
        alpha = xx
        param.fnumber = 1/2/np.tan(alpha)

    fNum = param.fnumber


    t0 = param.t0
    x = x.reshape((-1,1), order = 'F')
    z = z.reshape((-1,1), order = 'F')

    #%----
    #% Migration - diffraction summation (Delay & Sum, DAS)
    #%----

    #%-- TX distances
    #% For a given location [x(k),z(k)], one has:
    #% dTX(k) = min(delaysTXi*c + sqrt((xTi-x(k)).^2 + (zTi-z(k)).^2));
    #% In a compact matrix form, this yields:
    if param.passive:
        dTX = np.zeros_like(x)
    elif not useVirtualSource:
        # OLD version of computing the transmission delays
        #% For a given location [x(k),z(k)], one has:
        #% dTX(k) = min(delaysTXi*c + sqrt((xTi-x(k)).^2 + (zTi-z(k)).^2));
        #% In a compact matrix form, this yields:
        dTX = np.min(delaysTXi*c + np.sqrt((xTi-x)**2 + (zTi-z)**2), 1).reshape((-1,1))

    else:
        WasTransmitting = np.logical_not(np.isnan(delaysTX))
        assert np.sum(np.abs(np.diff(WasTransmitting)))<3, 'Multiple transmitting sub-apertures are not allowed during beamforming.'
        nTX = np.count_nonzero(WasTransmitting); # number of transmitting elements


        #-- TX distances
        if nTX==1:
            # Only one element was active
            dTX = np.hypot(xe[WasTransmitting][0]-x,ze[WasTransmitting][0]-z) + delaysTX[WasTransmitting][0]*c

        elif nTX<3:
            raise ValueError('If not 1, the number of neighboring transmitting elements must be at least 3.')
        
        else:
            #-- We create a virtual transducer to calculate the TX distances.
            xv,zv = vxdcr(xe[WasTransmitting],ze[WasTransmitting], delaysTX[WasTransmitting],c);
            dzv = diff2(xv,zv)

            # Distances between the (x,z) points and the lines normal to the
            # virtual transducer at (xv,zv)
            Dn = np.abs(x-xv + dzv*(z-zv))/np.hypot(1,dzv);
            # idx contains the element numbers that give the smallest Dn_s
            idx = np.argmin(Dn,1)
            
            InterpMethod = 'nearest';

            if InterpMethod == 'nearest':
                dTX = np.hypot(xv[idx].reshape((-1,1)) - x, zv[idx].reshape((-1,1))  -z)
            elif InterpMethod == 'parabolic':
                N = x.size
                dTX = np.hypot(xv.reshape((1,-1))-x,zv.reshape((1,-1))-z);
                dTX[:,1:nc] = scipy.signal.convolve2d(dTX, np.array([1, 0, 1]).reshape((1,-1)),'valid')/2 - \
                    scipy.signal.convolve2d(dTX,np.array([1, 0, -1]).reshape((1,-1)),'valid')* \
                    scipy.signal.convolve2d(Dn,np.array([1, 0, -1]).reshape((1,-1)),'valid')/ \
                scipy.signal.convolve2d(Dn,np.array([2, -4, 2]).reshape((1,-1)),'valid')

                dTX = dTX.flatten()
                dTX = dTX[np.arange(N) + idx *N ]
            else:
                raise ValueError("Incorrect interpolation method on dTX. Valid options: [nearest | parabolic]")



    #%-- RX distances
    #% For a given location [x(k),z(k)], one has:
    #% dRX(k) = sqrt((xT-x(k)).^2 + (zT-z(k)).^2);
    #% In a compact matrix form, this yields:
    dxT = x-xe
    dzT = z-ze
    dRX = np.sqrt(dxT**2 + dzT**2)

    #%-- Travel times
    tau = (dTX+dRX)/c

    #%-- Corresponding fast-time indices
    idxt = (tau-t0.reshape((1, -1)))*fs
    idxt = idxt.astype(np.float64) #% in case tau is in single precision

    #%-- In-range indices:
    method = method.lower()  
    if method == 'nearest':
        I = np.logical_and(idxt>=0, idxt<=nl - 1)
    elif method == 'linear':
        I = np.logical_and(idxt>=0, idxt<=nl -2 )
    elif method == 'quadratic':
        I = np.logical_and(idxt>=0, idxt<=nl -3 )
    elif method == 'lanczos3':
        I = np.logical_and(idxt>=0, idxt<=nl -3 )
    elif method == '5points':
        I = np.logical_and(idxt>=0, idxt<=nl -3 )
    elif method == 'lanczos5':
        I = np.logical_and(idxt>=0, idxt<=nl -4 )
    else:
        raise ValueError('Unknown interpolation method: %s' % method)

    #%-- Aperture (using the f-number):
    if fNum>0:
        if not isRXangle:
            if np.isfinite(RadiusOfCurvature):
                #% -- for a convex array
                Iaperture = np.abs(np.arcsin(dxT/dRX)-THe)<=np.arctan(1/2/fNum)
            else:
                #% -- for a linear array
                #% For a given location [x(k),z(k)], one has:
                #% Iaperture = abs(xT-x(k)) <= z(k)/2/fNum;
                Iaperture = np.abs(dxT)<=(z/2/fNum)
                #% Iaperture = mean(abs(dxT),1)<=(z/2/fNum);


        else: #% [Advanced option for vector Doppler: Reception angle]
            #% -- ONLY for a rectilinear array
            #% For a given location [x(k),z(k)], one has:
            #% Iaperture = abs(xT-x(k)-z(k)*tanRX(k)) <= z(k)/cosRX(k)/2/fNum;
            Iaperture = np.abs(dxT-z.tanRX.flatten())<=(z/cosRX.flatten()/2/fNum)

        I = np.logical_and(I,Iaperture)

    #I = np.asfortranarray(I)
    #% subscripts to linear indices (instead of using SUB2IND)
    idx_matrix = idxt + np.arange(nc).reshape((1, -1))*nl
    j, i = np.where(I.T) # GB: This weird inverion is to make sure that the ordering is consistent with matlab
    idx = np.take(idx_matrix, np.ravel_multi_index([i,j], I.shape)) 
    tau_I = np.take(tau, np.ravel_multi_index([i,j], I.shape))

    if method != 'nearest':
        idxf = np.floor(idx).astype(int)
        idx = idx-idxf


    #%-- Let's fill in the sparse DAS matrix
    if method == 'nearest':
        j = np.round(idx).astype(int)
        s = np.ones_like(i)
        n_repeat = 1
            
    elif method == 'linear': #%-- Linear interpolation (2-point method)            
        #%-- DAS matrix
        j = np.concatenate([idxf, idxf+1])
        s = np.concatenate([-idx+1,  idx])
        n_repeat = 2
    elif method == 'quadratic': #%-- quadratic interpolation (3-point method)

        #%-- DAS matrix
        j = np.concatenate([idxf, idxf+1, idxf+2])
        s = np.concatenate([(idx-1)*(idx-2)/2,
            -idx*(idx-2),
            idx*(idx-1)/2])        
        n_repeat = 3
    elif method ==  'lanczos3': #%-- 3-lobe Lanczos interpolation (4-point method)
        j = np.concatenate([idxf-1, idxf, idxf+1, idxf+2])
        s = np.concatenate([np.sinc(idx+1)*np.sinc((idx+1)/2),
            np.sinc(idx)*np.sinc(idx/2),
            np.sinc(idx-1)*np.sinc((idx-1)/2),
            np.sinc(idx-2)*
            np.sinc((idx-2)/2)])
        n_repeat = 4
    elif method == '5points': #%-- 5-point least-squares parabolic interpolation  
        j =  np.concatenate([idxf-2, idxf-1, idxf, idxf+1, idxf+2])
        idx2 = idx**2
        s *= np.concatenate([1/7*idx2-1/5*idx-3/35,
            -1/14*idx2-1/10*idx+12/35,
            -1/7*idx2+17/35,
            -1/14*idx2+1/10*idx+12/35,
            1/7*idx2+1/5*idx-3/35])
        n_repeat = 5
    elif method == 'lanczos5': #%-- 5-lobe Lanczos interpolation (6-point method)
        j = np.concatenate([idxf-2, idxf-1, idxf, idxf+1, idxf+2, idxf+3])
        s = np.concatenate([np.sinc(idx+2)*np.sinc((idx+2)/2),
            np.sinc(idx+1)*np.sinc((idx+1)/2),
            np.sinc(idx)*np.sinc(idx/2),
            np.sinc(idx-1)*np.sinc((idx-1)/2),
            np.sinc(idx-2)*np.sinc((idx-2)/2),
            np.sinc(idx-3)*np.sinc((idx-3)/2)])
        n_repeat = 6

    i = np.tile(i, n_repeat)
    if isIQ:
        s = np.exp(1j*wc*np.tile(tau_I, n_repeat)) * s

    if utils.isfield(param,'TransposeDASMatrix'):
            #% -- DASMTX has been called by the function DAS --
            #%
            #% The smallest DAS matrix (in terms of memory) is returned.
            #% (Matlab stores sparse matrices in compressed sparse column format)
            if len(x)>nl*nc:
                param.TransposeDASMatrix = False
                M = scipy.sparse.coo_matrix((s,(i,j)), shape = (len(x),nl*nc))
                #% M is a [numel(x)]-by-[nl*nc] sparse matrix
            else:
                param.TransposeDASMatrix = True
                M = scipy.sparse.coo_matrix((s,(j,i)),shape = (nl*nc, len(x)))
                #% M is a [nl*nc]-by-[numel(x)] sparse matrix
    else:
        M = scipy.sparse.coo_matrix((s,(i,j)), shape= (len(x),nl*nc))
            #% M is a [numel(x)]-by-[nl*nc] sparse matrix
    return M #TODO: maybe use another type of sparse matrix will be more efficient


class DasMTX:
    def __init__(self, SIG, x, z, *varargin):
        self.M = dasmtx(SIG, x, z, *varargin)
        self.shape = x.shape
    def __apply__(self, SIG, axis ):
        return (self.M @ SIG.flatten(order = 'F')).reshape(self.shape, order = 'F')


# Function to calculate the first derivative of y with respect to x using a second-order difference method
def diff2(x, y):
    m = len(y)
    dx = np.diff(x)
    dy = np.zeros_like(y)

    # y'(x_1)
    dy[0] = (1 / dx[0] + 1 / dx[1]) * (y[1] - y[0]) + dx[0] / (dx[0] * dx[1] + dx[1]**2) * (y[0] - y[2])

    # y'(x_m)
    dy[-1] = (1 / dx[-2] + 1 / dx[-1]) * (y[-1] - y[-2]) + dx[-1] / (dx[-1] * dx[-2] + dx[-2]**2) * (y[-3] - y[-1])

    # y'(x_i) (i > 1 & i < m)
    dx1 = dx[:-1]
    dx2 = dx[1:]
    y1 = y[:-2]
    y2 = y[1:-1]
    y3 = y[2:]
    dy[1:-1] = 1 / (dx1 * dx2 * (dx1 + dx2)) * (-dx2**2 * y1 + (dx2**2 - dx1**2) * y2 + dx1**2 * y3)

    return dy

def vxdcr(xe, ze, delaysTX, c):
    # Virtual transducer (XDCR)
    T2 = delaysTX**2
    dT2dxe = diff2(xe, T2)
    err = np.min(np.diff(xe)) * 1e-3
    dzedxe = diff2(xe, ze)

    # Virtual transducer positions
    if np.all(ze == 0):
        # For a linear transducer
        xV = xe - 0.5 * c**2 * dT2dxe
        zV = -np.sqrt(np.abs(c**2 * T2 - (xV - xe)**2))
    else:
        # For a convex array
        xV = xe.copy()
        tmp = np.sqrt(np.abs(c**2 * T2 - (xV - xe)**2))

        def myfun(xv, k):
            return (xe[k] - xv) + tmp[k] * dzedxe[k] - 0.5 * c**2 * dT2dxe[k]

        for k in range(len(xV)):
            xV[k] = scipy.optimize.fsolve(myfun, xV[k], args=(k,), xtol=err)[0]
        zV = ze - tmp

    # Validate the virtual transducer
    test = np.all(np.diff(xV) > 0) and np.all((c**2 * T2 - (xV - xe)**2) > -err)
    if not test:
        raise ValueError("The delays do not allow a virtual transducer to be calculated.")

    # Check if the virtual transducer is convex or concave
    idx = np.array([list(i) for i in itertools.combinations(range(len(xV)), 3)])
    lambda_vals = (xV[idx[:, 2]] - xV[idx[:, 1]]) / (xV[idx[:, 0]] - xV[idx[:, 1]])
    alst1 = np.abs(lambda_vals) < 1
    tmp = zV[idx[alst1, 2]] - (lambda_vals[alst1] * zV[idx[alst1, 0]] + (1 - lambda_vals[alst1]) * zV[idx[alst1, 1]])
    test = max(np.count_nonzero(tmp > -np.finfo(float).eps),
               np.count_nonzero(tmp < np.finfo(float).eps)) / len(tmp) > 0.999
    if not test:
        print("Warning: The virtual transducer is not convex or concave.")

    return xV, zV