import numpy as np, logging, typing
from . import utils
def impolgrid(siz : typing.Union[int, np.ndarray, list ], zmax : float, width:float, param : utils.Param =None):
    """
    %IMPOLGRID   Polar-type grid for ultrasound images
    %   IMPOLGRID returns a polar-type (fan-type) grid expressed in Cartesian
    %   coordinates. This is a "natural" grid (before scan-conversion) used
    %   when beamforming signals obtained with a cardiac phased array or a
    %   convex array.
    %
    %   [X,Z] = IMPOLGRID(SIZ,ZMAX,WIDTH,PARAM) returns the X,Z coordinates of
    %   the fan-type grid of size SIZ and angular width WIDTH (in rad) for a
    %   phased array described by PARAM. The maximal Z (maximal depth) is ZMAX.
    %
    %   [X,Z] = IMPOLGRID(SIZ,ZMAX,PARAM) returns the X,Z coordinates of
    %   the fan-type grid of size SIZ and angular width WIDTH (in rad) for a
    %   convex array described by PARAM. For a convex array, PARAM.radius is
    %   not Inf. The maximal Z (maximal depth) is ZMAX.
    %
    %   If SIZ is a scalar M, then the size of the grid is [M,M].
    %
    %   [X,Z,Z0] = IMPOLGRID(...) also returns the z-coordinate of the grid
    %   origin. Note that X0 = 0.
    %
    %   Units: X,Z,Z0 are in m. WIDTH must be in rad.
    %
    %   PARAM is a structure which must contain the following fields:
    %   ------------------------------------------------------------
    %   1) PARAM.pitch: pitch of the array (in m, REQUIRED)
    %   2) PARAM.Nelements: number of elements in the transducer array (REQUIRED)
    %   3) PARAM.radius: radius of curvature (in m, default = Inf, linear array)
    %
    %
    %   Examples:
    %   --------
    %   %-- Generate a focused pressure field with a phased-array transducer
    %   % Phased-array @ 2.7 MHz:
    %   param = getparam('P4-2v');
    %   % Focus position:
    %   xf = 2e-2; zf = 5e-2;
    %   % TX time delays:
    %   dels = txdelay(xf,zf,param);
    %   % 60-degrees wide grid:
    %   [x,z] = impolgrid([100 50],10e-2,pi/3,param);
    %   % RMS pressure field:
    %   P = pfield(x,z,dels,param);
    %   % Scatter plot of the pressure field:
    %   figure
    %   scatter(x(:)*1e2,z(:)*1e2,5,20*log10(P(:)/max(P(:))),'filled')
    %   colormap jet, axis equal ij tight
    %   xlabel('cm'), ylabel('cm')
    %   caxis([-20 0])
    %   c = colorbar;
    %   c.YTickLabel{end} = '0 dB';
    %   % Image of the pressure field:
    %   figure
    %   pcolor(x*1e2,z*1e2,20*log10(P/max(P(:))))
    %   shading interp
    %   colormap hot, axis equal ij tight
    %   xlabel('[cm]'), ylabel('[cm]')
    %   caxis([-20 0])
    %   c = colorbar;
    %   c.YTickLabel{end} = '0 dB';
    %
    %
    %   This function is part of MUST (Matlab UltraSound Toolbox).
    %   MUST (c) 2020 Damien Garcia, LGPL-3.0-or-later
    %
    %   See also DAS, DASMTX, PFIELD.
    %
    %   -- Damien Garcia -- 2020/05, last update: 2022/03/30
    %   website: <a
    %   href="matlab:web('https://www.biomecardio.com')">www.BiomeCardio.com</a>
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