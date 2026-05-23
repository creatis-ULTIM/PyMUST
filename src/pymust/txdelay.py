"""
.. note:: Documentation auto-generated with Claude (claude-sonnet-4-6).
"""
from . import utils
import numpy as np

def txdelayCircular(param: utils.Param, tilt: float, width: float) -> np.ndarray:
    return txdelay(param, tilt, width)

def txdelayPlane(param: utils.Param, tilt: float) -> np.ndarray:
    return txdelay(param, tilt)

def txdelayFocused(param: utils.Param, x: float, y: float) -> np.ndarray:
    return txdelay(x, y, param)

def txdelay(*args):
    """Compute transmit time delays for focused, plane, or circular wavefronts.

    Supports three calling conventions (variable-argument style matching the
    original MATLAB interface):

    - ``txdelay(x0, z0, param)`` — focused beam at ``(x0, z0)``.
      Negative *z0* creates a virtual source (diverging wave).
    - ``txdelay(param, tilt)`` — tilted plane wave at angle *tilt* (rad).
    - ``txdelay(param, tilt, width)`` — circular wave sector defined by
      *tilt* and angular *width* (rad). Linear arrays only.

    *x0*, *z0*, *tilt*, and *width* may be vectors; *delays* is then a
    matrix with one row per delay law.

    Parameters
    ----------
    param : utils.Param
        Transducer parameter structure with the following fields:

        - ``pitch``     : element pitch (m, required)
        - ``Nelements`` : number of elements (required)
        - ``radius``    : radius of curvature (m, default ``np.inf``)
        - ``c``         : speed of sound (m/s, default 1540)
    x0, z0 : float or array-like
        Focus coordinates (m); used in the focused-beam form.
    tilt : float or array-like
        Plane/circular-wave tilt angle about the Y-axis (rad, trigonometric
        direction).
    width : float or array-like
        Circular-wave sector angular width (rad); linear arrays only.

    Returns
    -------
    delays : np.ndarray
        TX time delays (s).  Shape is ``(1, Nelements)`` for a single law or
        ``(N, Nelements)`` for *N* simultaneous laws.

    Examples
    --------
    Focused beam at (2 cm, 5 cm):

    >>> from pymust import getparam, txdelay
    >>> param = getparam('P4-2v')
    >>> dels = txdelay(2e-2, 5e-2, param)

    Tilted plane wave at 10°:

    >>> import numpy as np
    >>> dels = txdelay(param, np.deg2rad(10))

    Notes
    -----
    Also available as convenience wrappers :func:`txdelayFocused`,
    :func:`txdelayPlane`, and :func:`txdelayCircular`.

    Translated from the MATLAB MUST toolbox (Damien Garcia, 2015–2022).
    MUST (c) 2020 Damien Garcia, LGPL-3.0-or-later

    See Also
    --------
    pfield, simus, dasmtx, getparam, impolgrid
    """

    #%-- Check the input arguments
    if len(args) ==2: # % Plane wave: TXDELAY(param,tilt)
        param = args[0]
        option = 'Plane Wave'
    elif len(args) == 3:
        if isinstance(args[2], utils.Param): #% Origo: TXDELAY(x0,z0,param)
            param = args[2]
            option = 'Origo'
        else:#  % Circular wave: TXDELAY(param,tilt,width)
            param = args[0]
            option = 'Circular Wave'
    else:
        ValueError('Wrong input arguments.')

    assert isinstance(param, utils.Param),'Wrong input arguments. PARAM must be a structure.'

    #%-- Number of elements
    if utils.isfield(param,'Nelements'):
        N = param.Nelements
    else:
        raise ValueError('The number of elements (PARAM.Nelements) is required.')

    #%-- Pitch (in m)
    if not utils.isfield(param,'pitch'):
        raise ValueError('A pitch value (PARAM.pitch) is required.')

    #%-- Longitudinal velocity (in m/s)
    if not utils.isfield(param,'c'):
        param.c = 1540

    c = param.c

    #%-- Radius of curvature (in m)
    #% for a convex array
    if not utils.isfield(param,'radius'):
        param.radius = np.inf # % default = linear array

    R = param.radius
    isLINEAR = np.isinf(R)




    #%-- Positions of the transducer elements
    x, z, THe, h= param.getElementPositions()

    if option == 'Plane Wave':
        tilt = np.array(args[1]).reshape((-1, 1)) # Check if it is not a vector
        assert np.all(np.abs(tilt)<np.pi/2), 'The tilt angles must verify |tilt| < pi/2'
        if isLINEAR:
            delays = x*np.sin(tilt)/c
        else:
            #% we have a CONVEX ARRAY
            
            #% intersection point between the wavefront and the transducer
            xn = R*np.sin(tilt)
            zn = R*np.cos(tilt)-h

            #% Note:
            #% Equation of the line tangent to the transducer at (xn,zn):
            #% X = -xn/(zn+h)*(X-xn) + zn
            
            #% distances between this line and the elements (x,z)
            d = np.abs(z+xn/(zn+h)*x-xn**2/(zn+h)-zn)/ \
                np.sqrt(1+xn**2/(zn+h)**2)
            delays = -d/c
    #%-----
    elif option == 'Origo':
        x0 = np.array(args[0]).reshape((-1, 1))
        z0 = np.array(args[1]).reshape((-1, 1))
        assert x0.shape == z0.shape, 'X0 and Z0 must have the same length.'
        delays = np.sqrt((x-x0)**2 + (z-z0)**2)/c
        if isLINEAR:
            delays = -delays*np.sign(z0)
        elif np.sqrt(x0**2 + (R-z0)**2)<R:
            delays = -delays
    #%-----
    elif option == 'Circular Wave':
        assert isLINEAR,'The syntax "TXDELAY(PARAM,TILT,WIDTH)" is not available for a convex array.'
        tilt = np.array(args[1]).reshape((-1, 1))
        width = np.array(args[2]).reshape((-1, 1))
        assert tilt.shape == width.shape, 'TILT and WIDTH must have the same length.'
        assert np.all(np.logical_and(width>0, width<np.pi)), 'The width angles must verify width > 0 and width < pi'
        L = (N-1)*param.pitch
        #%-- Origo
        x0,z0 = angles2origo(L,tilt,width)
        #%--
        delays = np.sqrt((x-x0)**2 + z0**2)/c
        delays = -delays*np.sign(z0)
    delays = delays-np.min(delays,-1).reshape((-1, 1))

    param.TXdelay = delays
    return delays

def angles2origo(L,tilt,width):
    #% Origo (virtual source) from the tilt and width angles
    tilt = np.mod(-tilt+np.pi/2,2*np.pi)-np.pi/2
    SignCorrection = np.ones(tilt.shape)
    idx = np.abs(tilt)>np.pi/2
    tilt[idx] = np.pi-tilt[idx]
    SignCorrection[idx] = -1
    z0 = SignCorrection*L/(np.tan(tilt-width/2)-np.tan(tilt+width/2))
    x0 = SignCorrection*z0*np.tan(width/2-tilt)+L/2
    return x0, z0
