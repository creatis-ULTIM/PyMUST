"""
.. note:: Documentation auto-generated with Claude (claude-sonnet-4-6).
"""
import numpy as np, logging, typing
from . import utils

def impolgrid(siz: typing.Union[int, np.ndarray, list], zmax: float, width: float, param: utils.Param = None):
    """Return a polar-type (fan-type) Cartesian grid for ultrasound imaging.

    Generates the "natural" polar grid used when beamforming data from a
    phased array or convex array (before scan-conversion).

    Parameters
    ----------
    siz : int or array-like
        Grid size ``[nrows, ncols]``.  A scalar *M* produces an *M×M* grid.
    zmax : float
        Maximum depth (m).
    width : float
        Angular width of the fan sector (rad). Ignored for convex arrays
        (``PARAM.radius < inf``); the angular extent is derived from the array
        geometry in that case.
    param : utils.Param
        Transducer parameter structure with the following fields:

        - ``pitch``     : element pitch (m, required)
        - ``Nelements`` : number of array elements (required)
        - ``radius``    : radius of curvature (m, default ``np.inf`` for a
          linear array)

        For a convex array call ``impolgrid(siz, zmax, param)`` (omit *width*).

    Returns
    -------
    x : np.ndarray
        x-coordinates of grid points (m), shape ``siz``.
    z : np.ndarray
        z-coordinates of grid points (m), shape ``siz``.

    Examples
    --------
    60-degree fan grid for a phased array:

    >>> from pymust import getparam, txdelay, pfield, impolgrid
    >>> import numpy as np
    >>> param = getparam('P4-2v')
    >>> dels = txdelay(2e-2, 5e-2, param)
    >>> x, z = impolgrid([100, 50], 10e-2, np.pi/3, param)
    >>> P = pfield(x, np.zeros_like(x), z, dels, param)

    Notes
    -----
    Translated from the MATLAB MUST toolbox (Damien Garcia, 2020–2022).
    MUST (c) 2020 Damien Garcia, LGPL-3.0-or-later

    See Also
    --------
    dasmtx, pfield, txdelay
    """
    noWidth = False

    #GB: Change the arguments names... this is nonpythonic, but keeping consistent with matlab implementation
    if param is None:
        param = width
        noWidth = True


    assert isinstance(siz, int) or len(siz)==1 or len(siz)==2,'SIZ must be [M,N] or M.'
    if isinstance(siz, int):
        siz = np.array([siz, siz])

    assert np.all(siz>0) and np.issubdtype(siz.dtype, np.integer), 'SIZ components must be positive integers.'

    assert np.isscalar(zmax) and zmax>0, 'ZMAX must be a positive scalar.'

    assert isinstance(param, utils.Param),'PARAM must be a structure.'

    #%-- Pitch (in m)
    if not utils.isfield(param,'pitch'):
        raise ValueError('A pitch value (PARAM.pitch) is required.')
    p = param.pitch

    #%-- Number of elements
    if utils.isfield(param,'Nelements'):
        N = param.Nelements
    else:
        raise ValueError('The number of elements (PARAM.Nelements) is required.')


    #%-- Radius of curvature (in m)
    #% for a convex array
    if not utils.isfield(param,'radius'):
        param.radius = np.inf #% default = linear array

    R = param.radius
    isLINEAR = np.isinf(R)

    if not isLINEAR and not noWidth:
        logging.warning('MUST:impolgrid', 'The parameter WIDTH is ignored with a convex array.')

    #%-- Origo (x0,z0)
    #% x0 = 0;
    if isLINEAR:
        L = (N-1)*p# % array width
        #% z0 = -L/2*(1+cos(width))/sin(width); % (old version)
        z0 = 0
    else:
        L = 2*R*np.sin(np.arcsin(p/2/R)*(N-1)) # % chord length
        d = np.sqrt(R**2-L**2/4) # % apothem
        #% https://en.wikipedia.org/wiki/Circular_segment
        z0 = -d


    #%-- Image polar grid
    if isLINEAR:
        R = np.hypot(L/2,z0)
        th,r = np.meshgrid( 
            np.linspace(width/2,-width/2,siz[1])+np.pi/2,
            np.linspace(R+p,-z0+zmax,siz[0]))
        x,z = pol2cart(th,r)
    else:
        th,r = np.meshgrid(
            np.linspace(np.arctan2(L/2,d),np.arctan2(-L/2,d),siz[1])+np.pi/2,
            np.linspace(R+p,-z0+zmax,siz[0]))
        x,z = pol2cart(th,r)

    z = z+z0
    return x, z

def pol2cart(th, r):
    return r*np.cos(th), r*np.sin(th)
