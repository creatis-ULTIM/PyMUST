import scipy, numpy as np, typing, logging
from . import utils


def smoothn(y : np.ndarray,
            W : typing.Optional[np.ndarray] = None, 
            S  : typing.Optional[float] = None,  
            axis : typing.Optional[np.ndarray] = None, # Use this to specify the axis indicating multicomponent data
            TolZ : typing.Optional[float] = 1e-3,
            MaxIter : typing.Optional[int] = 100,
            Initial : typing.Optional[np.ndarray] = None,
            Spacing : typing.Optional[np.ndarray] = None,
            Order : typing.Optional[int] = 2,   
            Weight : str = 'bisquare',
            isrobust : bool = False ):
    """
    %SMOOTHN Robust spline smoothing for 1-D to N-D data.
    %   SMOOTHN provides a fast, automatized and robust discretized spline
    %   smoothing for data of arbitrary dimension.
    %
    %   Z = SMOOTHN(Y) automatically smoothes the uniformly-sampled array Y. Y
    %   can be any N-D noisy array (time series, images, 3D data,...). Non
    %   finite data (NaN or Inf) are treated as missing values.
    %
    %   Z = SMOOTHN(Y,S) smoothes the array Y using the smoothing parameter S.
    %   S must be a real positive scalar. The larger S is, the smoother the
    %   output will be. If the smoothing parameter S is omitted (see previous
    %   option) or empty (i.e. S = []), it is automatically determined by
    %   minimizing the generalized cross-validation (GCV) score.
    %
    %   Z = SMOOTHN(Y,W) or Z = SMOOTHN(Y,W,S) smoothes Y using a weighting
    %   array W of positive values, which must have the same size as Y. Note
    %   that a nil weight corresponds to a missing value.
    %
    %   If you want to smooth a vector field or multicomponent data, Y must be
    %   a cell array. For example, if you need to smooth a 3-D vectorial flow
    %   (Vx,Vy,Vz), use Y = {Vx,Vy,Vz}. The output Z is also a cell array which
    %   contains the smoothed components. See examples 5 to 8 below.
    %
    %   [Z,S] = SMOOTHN(...) also returns the calculated value for the
    %   smoothness parameter S.
    %
    %
    %   1) ROBUST smoothing
    %   -------------------
    %   Z = SMOOTHN(...,'robust') carries out a robust smoothing that minimizes
    %   the influence of outlying data.
    %
    %   An iteration process is used with the 'ROBUST' option, or in the
    %   presence of weighted and/or missing values. Z = SMOOTHN(...,OPTIONS)
    %   smoothes with the termination parameters specified in the structure
    %   OPTIONS. OPTIONS is a structure of optional parameters that change the
    %   default smoothing properties. It must be the last input argument.
    %   ---
    %   The structure OPTIONS can contain the following fields:
    %       -----------------
    %       OPTIONS.TolZ:       Termination tolerance on Z (default = 1e-3),
    %                           OPTIONS.TolZ must be in ]0,1[
    %       OPTIONS.MaxIter:    Maximum number of iterations allowed
    %                           (default = 100)
    %       OPTIONS.Initial:    Initial value for the iterative process
    %                           (default = original data, Y)
    %       OPTIONS.Weight:     Weight function for robust smoothing:
    %                           'bisquare' (default), 'talworth' or 'cauchy'
    %       -----------------
    %
    %   [Z,S,EXITFLAG] = SMOOTHN(...) returns a boolean value EXITFLAG that
    %   describes the exit condition of SMOOTHN:
    %       1       SMOOTHN converged.
    %       0       Maximum number of iterations was reached.
    %
    %
    %   2) Different spacing increments
    %   -------------------------------
    %   SMOOTHN, by default, assumes that the spacing increments are constant
    %   and equal in all the directions (i.e. dx = dy = dz = ...). This means
    %   that the smoothness parameter is also similar for each direction. If
    %   the increments differ from one direction to the other, it can be useful
    %   to adapt these smoothness parameters. You can thus use the following
    %   field in OPTIONS:
    %       OPTIONS.Spacing = [d1 d2 d3...],
    %   where dI represents the spacing between points in the Ith dimension.
    %
    %   Important note: d1 is the spacing increment for the first
    %   non-singleton dimension (i.e. the vertical direction for matrices).
    %
    %
    %   3) REFERENCES (please refer to the two following papers)
    %   ------------- 
    %   1) Garcia D, Robust smoothing of gridded data in one and higher
    %   dimensions with missing values. Computational Statistics & Data
    %   Analysis, 2010;54:1167-1178. 
    %   <a
    %   href="matlab:web('http://www.biomecardio.com/publis/csda10.pdf')">download PDF</a>
    %   2) Garcia D, A fast all-in-one method for automated post-processing of
    %   PIV data. Exp Fluids, 2011;50:1247-1259.
    %   <a
    %   href="matlab:web('http://www.biomecardio.com/publis/expfluids11.pdf')">download PDF</a>
    %
    %
    """
    ##%% Check input arguments

    # Test & prepare the variables
    #---
    # y = array to be smoothed
    if isinstance(y, list) or isinstance(y, tuple):
        if not axis is None:
            raise ValueError('If Y is a list or tuple, it is considered a multivariate array and the axis parameter must be None')
        y = np.array(y)
        axis = [0]

    origShapeY = y.shape
    # If the axis is None, then it is a single array
    if axis is None:
         axis = [0]
         y =y[np.newaxis, :]


    
    # Axis must be a list
    if not  np.allclose(axis, np.arange(len(axis))):
        raise NotImplemented('Now, multivariate arrays must be in the first dimensions (ordered)')
    else:
         # Merge all the indices being multivariate array in the first component.
         y = y.reshape((-1, *[d for i, d in enumerate(y.shape) if i not in axis]))
    sizy = y[0].shape
    ny = y.shape[0]; # number of y components
    for i, yi in enumerate(y):
        assert np.all(np.equal(sizy,yi.shape)), 'Data arrays in Y must have the same size.'


    noe = np.prod(sizy)# % number of elements
    if noe==1:
        return y.reshape(origShapeY), [], True

    #---
    # Smoothness parameter and weights
    if W is None:
        W = np.ones(sizy)
    if S is None:
        isauto = True
    else:
        isauto = False
        assert (np.isscalar(S) and S>0), 'The smoothing parameter S must be a scalar >0'
        assert utils.isnumeric(S),'S must be a numeric scalar'

    assert utils.isnumeric(W),'W must be a numeric array'
    assert np.all(np.equal(W.shape,sizy)), 'Arrays for data and weights (Y and W) must have same size.'
    
    #---
    # Field names in the structure OPTIONS

    assert utils.isnumeric(MaxIter) and np.isscalar(MaxIter) and MaxIter>=1 and MaxIter==np.round(MaxIter), 'OPTIONS.MaxIter must be an integer >=1'
    MaxIter = int(MaxIter)

    assert utils.isnumeric(TolZ) and np.isscalar(TolZ) and TolZ>0 and TolZ<1,'OPTIONS.TolZ must be in ]0,1['
    #---
    # "Initial Guess" criterion
    if Initial is None:
        isinitial = False
    else:
        isInitial = True
        z0 = Initial;
        z = z.reshape((-1, *[d for i, d in enumerate(origShapeY) if i not in axis]))
        assert np.all_equal(z.shape, y.shape), 'OPTIONS.Initial must contain a valid initial guess for Z'


    #%---
    # "Weight function" criterion (for robust smoothing)
    Weight = Weight.lower()
    assert Weight in ['bisquare','talworth','cauchy'], 'The weight function must be ''bisquare'', ''cauchy'' or '' talworth''.'
    
    #---
    # "Order" criterion (by default m = 2)
    # Note: m = 0 is of course not recommended!

    m = Order
    assert m in [0,1,2], 'The order (OPTIONS.order) must be 0, 1 or 2.'
    
    #---
    # "Spacing" criterion
    d = len(y[0].shape)
    if Spacing is None:
        dI = np.ones(d) #
    else:
        dI = Spacing;
        assert utils.isnumeric(dI) and len(dI) == d, 'A valid spacing (OPTIONS.Spacing) must be chosen'

    dI = dI/max(dI);
    #---
    # Weights. Zero weights are assigned to not finite values (Inf or NaN),
    # (Inf/NaN values = missing data).
    IsFinite = np.all(np.isfinite(y), axis = 0)
    nof = np.count_nonzero(IsFinite); #% number of finite elements
    W = W*IsFinite;
    assert np.all(W>=0,) & np.all(np.isfinite(W)), 'Weights must all be finite and >=0'
    # W = W/max(W(:));
    #---
    # Weighted or missing data?
    isweighted = np.any(W !=1)

    #---
    # Automatic smoothing?


    #% Create the Lambda tensor
    #---
    # Lambda contains the eingenvalues of the difference matrix used in this
    # penalized least squares process (see CSDA paper for details)
    Lambda = np.zeros(sizy);
    for i in range(d):
        siz0 = np.ones((1,d))
        siz0[0,i] = sizy[i]
        Lambda = Lambda + (2-2*np.cos(np.pi*(np.arange(sizy[i],dtype = float).reshape([1 if i != j else sizy[i] for j in range(d)]))/sizy[i]))/dI[i]**2

    if not isauto:
        Gamma = 1/(1+S*Lambda**m)
    #% Upper and lower bound for the smoothness parameter
    # The average leverage (h) is by definition in [0 1]. Weak smoothing occurs
    # if h is close to 1, while over-smoothing appears when h is near 0. Upper
    # and lower bounds for h are given to avoid under- or over-smoothing. See
    # equation relating h to the smoothness parameter for m = 2 (Equation #12
    # in the referenced CSDA paper).
    N = np.sum(sizy!=1) # tensor rank of the y-array
    hMin = 1e-6
    hMax = 0.99
    if m==0: # Not recommended. For mathematical purpose only.
        sMinBnd = 1/hMax**(1/N)-1;
        sMaxBnd = 1/hMin**(1/N)-1;
    elif m==1:
        sMinBnd = (1/hMax**(2/N)-1)/4;
        sMaxBnd = (1/hMin**(2/N)-1)/4;
    elif m==2:
        sMinBnd = ((1+np.sqrt(1+8*hMax**(2/N)))/4/hMax**(2/N)**2-1)/16;
        sMaxBnd = (((1+np.sqrt(1+8*hMin**(2/N)))/4/hMin**(2/N))**2-1)/16;
    

    #% Initialize before iterating
    #---
    Wtot = W;
    #--- Initial conditions for z
    if isweighted:
        #--- With weighted/missing data
        # An initial guess is provided to ensure faster convergence. For that
        # purpose, a nearest neighbor interpolation followed by a coarse
        # smoothing are performed.
        #---
        if isinitial:#% an initial guess (z0) has been already given
            z = z0;
        else:
            z = InitialGuess(y,IsFinite)

    else:
        z = np.zeros_like(y)
    #---
    z0 = z;
    for i, _ in enumerate(y):
        y[i][np.logical_not(IsFinite)] = 0 #arbitrary values for missing y-data
    

    #---
    tol = 1;
    RobustIterativeProcess = True;
    RobustStep = 1;
    nit = 0;
    DCTy = np.zeros_like(y);
    #--- Error on p. Smoothness parameter s = 10^p
    errp = 0.1;

    #--- Relaxation factor RF: to speedup convergence
    RF = 1 + 0.75*isweighted;

    #%% Main iterative process
    #%---
    while RobustIterativeProcess:
        #--- "amount" of weights (see the function GCVscore)
        aow = np.sum(Wtot)/np.max(W)/noe; # 0 < aow <= 1
        #---
        while tol> TolZ and nit<MaxIter:
            nit = nit+1
            for i in range(ny):
                DCTy[i] = scipy.fft.dctn(Wtot*(y[i]-z[i])+z[i]);
            
            if isauto and float.is_integer(np.log2(nit)): 
                #---
                # The generalized cross-validation (GCV) method is used.
                # We seek the smoothing parameter S that minimizes the GCV
                # score i.e. S = Argmin(GCVscore).
                # Because this process is time-consuming, it is performed from
                # time to time (when the step number - nit - is a power of 2)
                #---
                x = scipy.optimize.fminbound(gcv,np.log10(sMinBnd),np.log10(sMaxBnd),
                                                    xtol = errp, 
                                                    args = (y, Wtot, IsFinite, Lambda, m,aow, nof, noe, ny, DCTy))
                S = 10**x
                # Need to update Gamma
                Gamma = 1 / (1 + S * Lambda ** m)
            for i in range(ny):
                z[i] = RF*scipy.fft.idctn(Gamma*DCTy[i]) + (1-RF)*z[i]
            
            
            # if no weighted/missing data => tol=0 (no iteration)
            tol = isweighted*np.linalg.norm( z0 - z)/np.linalg.norm(z0)
            z0 = z.copy() # re-initialization

        exitflag = nit< MaxIter
        if isrobust: #-- Robust Smoothing: iteratively re-weighted process
            ##--- average leverage
            h = 1;
            for k in range(N):
                if m==0: ## not recommended - only for numerical purpose
                    h0 = 1/(1+S/dI[k]**(2**m));
                elif  m==1:
                    h0 = 1/np.sqrt(1+4*S/dI[k]*(2**m));
                elif  m==2:
                    h0 = np.sqrt(1+16*S/dI[k]**(2**m));
                    h0 = np.sqrt(1+h0)/np.sqrt(2)/h0;
                h = h*h0;
            #%--- take robust weights into account
            Wtot = W*RobustWeights(y,z,IsFinite,h, Weight);
            
            #%--- re-initialize for another iterative weighted process
            isweighted = True;
            tol = 1;
            nit = 0; 
            #%---
            RobustStep = RobustStep+1
            RobustIterativeProcess = RobustStep<4  # 3 robust steps are enough.
        else:
            RobustIterativeProcess = False; # stop the whole process
        
    

   #% Warning messages
    #%---
    if isauto:
        if abs(np.log10(S)-np.log10(sMinBnd))<errp:
            logging.warning(f"smoothn:SLowerBound S =  {S:.3e} : the lower bound for S  has been reached. Specify S as an input variable.")
        elif abs(np.log10(S)-np.log10(sMaxBnd))<errp:
             logging.warning(f"smoothn:SUpperBound S =  {S:.3e} : the upper bound for S  has been reached. Specify S as an input variable.")
        
    if not exitflag:
       logging.warning('smoothn:MaxIter Maximum number of iterations ({MaxIter}) has been exceeded. Increase MaxIter option (MaxIter) or decrease TolZ (TolZ) value.')
    
    return z.reshape(origShapeY), S, exitflag




#% GCV score
#---
def gcv(p,  y, Wtot, IsFinite, Lambda, m,aow, nof, noe, ny, DCTy):
    # Search the smoothing parameter s that minimizes the GCV score
    #---
    s = 10**p;
    Gamma = 1/(1+s*Lambda**m);
    #--- RSS = Residual sum-of-squares
    RSS = 0;
    if aow>0.95: # aow = 1 means that all of the data are equally weighted
        # very much faster: does not require any inverse DCT
        for kk in range(ny):
            RSS = RSS + np.linalg.norm(DCTy[kk]*(Gamma-1))**2;
    else:
        # take account of the weights to calculate RSS:
        for kk in range(ny):
            yhat = scipy.fft.idctn(Gamma*DCTy[kk]);
            RSS = RSS + np.linalg.norm(np.sqrt(Wtot[IsFinite])* (y[kk][IsFinite]-yhat[IsFinite]))**2;
        #%---
    TrH = np.sum(Gamma)
    GCVscore = RSS/nof/(1-TrH/noe)**2
    return GCVscore



def RobustWeights(y,z,I,h,wstr):
    # One seeks the weights for robust smoothing...

    r =  y- z
    r = r.reshape((r.shape[0], -1))
    rI = r[:, I.flatten()]
    MMED = np.median(rI); # marginal median
    AD = np.linalg.norm(rI-MMED, axis = 0); # absolute deviation
    MAD = np.median(AD); # median absolute deviation

    #%-- Studentized residuals
    u = np.linalg.norm(r, axis = 0)/(1.4826*MAD)/np.sqrt(1-h); 
    u = u.reshape(I.shape)
    #u = u.reshape(I.shape)
    if wstr == 'cauchy':
        c = 2.385
        W = 1/(1+(u/c)**2); #% Cauchy weights
    elif wstr == 'talworth':
        c = 2.795
        W = u<c  # Talworth weights
    elif wstr == 'bisquare':
        c = 4.685
        W = (1-(u/c)**2)**2*((u/c)<1); # bisquare weights
    else:
        raise ValueError('A valid weighting function must be chosen')
    W[np.isnan(W)] = 0
    return W

"""
% NOTE:
% ----
% The RobustWeights subfunction looks complicated since we work with cell
% arrays. For better clarity, here is how it would look like without the
% use of cells. Much more readable, isn't it?
%
% function W = RobustWeights(y,z,I,h)
%     % weights for robust smoothing.
%     r = y-z; % residuals
%     MAD = median(abs(r(I)-median(r(I)))); % median absolute deviation
%     u = abs(r/(1.4826*MAD)/sqrt(1-h)); % studentized residuals
%     c = 4.685; W = (1-(u/c).^2).^2.*((u/c)<1); % bisquare weights
%     W(isnan(W)) = 0;
% end

"""


# Initial Guess with weighted/missing data
def InitialGuess(y,I):
    ny = y.shape[0]
    #%-- nearest neighbor interpolation (in case of missing values)
    if np.any(np.logical_not(I)):
        z = np.zeros_like(y)        
        for i in range(ny):
            _,L = scipy.ndimage.distance_transform_edt(np.logical_not(I), return_indices = True)
            z[i] = y[i]
            #z[i][np.logical_not(I)] = y[i][*L[:,np.logical_not(I)]]; #Use np take
            z[i][np.logical_not(I)] = np.take(y[i], L[:, np.logical_not(I)])
    else:
        z = y

    #-- coarse fast smoothing using one-tenth**d of the DCT coefficients
    z = scipy.fft.dctn(z, axes = np.arange(1, len(z.shape)))
    indices = np.indices(z.shape) / np.array(z.shape).reshape(([-1] + [1 for _ in range(z.ndim)])) #Normalised indices between 0 and 1
    maxIndex = np.max(indices, axis = 0)
    z[maxIndex >.1] = 0 # np.power(.1, 1/z.ndim) so that it has constant
    z = scipy.fft.idctn(z,  axes = np.arange(1, len(z.shape)))
    return z
