import numpy as np,scipy, logging
import typing

def wfilt(SIG :np.ndarray, method : str, n: int) -> np.ndarray :
    """
    %WFILT   Wall filtering (or clutter filtering)
    %   fSIG = WFILT(SIG,METHOD,N) high-pass (wall) filters the RF or I/Q
    %   signals stored in the 3-D array SIG for Doppler imaging.
    %    
    %   The first dimension of SIG (i.e. each column) corresponds to a single
    %   RF or I/Q signal over (fast-) time, with the first column corresponding
    %   to the first transducer element. The third dimension corresponds to the
    %   slow-time axis.
    %
    %   Three methods are available.
    %   METHOD can be one of the following (case insensitive):
    %
    %   1) 'poly' - Least-squares (Nth degree) polynomial regression.
    %               Orthogonal Legendre polynomials are used. The fitting
    %               polynomial is removed from the original I/Q or RF data to
    %               keep the high-frequency components. N (>=0) represents the
    %               degree of the polynomials. The (slow-time) mean values are
    %               removed if N = 0 (the polynomials are reduced to
    %               constants).
    %   2) 'dct'  - Truncated DCT (Discrete Cosine Transform).
    %               Discrete cosine transforms (DCT) and inverse DCT are
    %               performed along the slow-time dimension. The signals are
    %               filtered by withdrawing the first N (>=1) components, i.e.
    %               those corresponding to the N lowest frequencies (with
    %               respect to slow-time).
    %   3) 'svd'  - Truncated SVD (Singular Value Decomposition).
    %               An SVD is carried out after a column arrangement of the
    %               slow-time dimension. The signals are filtered by
    %               withdrawing the top N singular vectors, i.e. those
    %               corresponding to the N greatest singular values.
    %   
    %
    %   This function is part of MUST (Matlab UltraSound Toolbox).
    %   MUST (c) 2020 Damien Garcia, LGPL-3.0-or-later
    %
    %   See also IQ2DOPPLER, RF2IQ.
    %
    %   -- Damien Garcia -- 2014/06, last update 2023/05/12
    %   website: <a
    %   href="matlab:web('https://www.biomecardio.com')">www.BiomeCardio.com</a>
    """
    logging.warning('NOTE GB: this code has not been tested!')

    #%-- Check the input arguments

    assert SIG.ndims ==3 and SIG.shape[2] >= 2,'SIG must be a 3-D array with SIG.shape[2]>=2';
    assert isinstance(n, int) and n >= 0, 'N must be a nonnegative integer.'

    siz0 = SIG.shape
    N = siz0[2]; # number of slow-time samples
    method = method.lower()

    if method == 'poly':
        #% ---------------------------------
        #% POLYNOMIAL REGRESSION WALL FILTER
        #% ---------------------------------

        assert N>n,'The packet length must be >N.'
        
        # If the degree is 0, the mean is removed.
        if n==0:
            return  SIG-np.mean(SIG,2);
    
        # GB TODO: use Legendre Matrix instead (more numerically stable and efficient)
        V = np.vander(np.linspace(0,1,N), n+1) # Vandermonde matrix
        A = np.eye(N) - V @ np.linalg.pinv(V) # Projection matrix
        # Multiply along the slow-time dimension
        SIG = np.einsum('ij,nkj->nki', A, SIG)
       

    elif method == 'dct':
        #% -------------------------------------
        #% DISCRETE COSINE TRANSFORM WALL FILTER
        #% -------------------------------------
        
        assert n>0, "N must be >0 with the ''dct'' method."
        assert N>=n,'The packet length must be >=N.'
        
        #% If the degree is 0, the mean is removed.
        if n==1:
            return SIG-np.mean(SIG,2);
        
        D = scipy.fft.dct(np.eye(N), norm='ortho', axis=0)[n:, :] #DCT matrix, only high frequencies
        D= D.T@D # Create the projection matrix
        #Multiply along the slow-time dimension
        SIG = np.einsum('ij,nkj->nki', D, SIG)        
        
    elif method == 'svd':
        #% ----------------------------------------
        #% SINGULAR VALUE DECOMPOSITION WALL FILTER
        #% ----------------------------------------
        
        assert n>0,'N must be >0 with the ''svd'' method.'
        assert N>=n,'The packet length must be >=N.'
        
        #% Each column represents a column-rearranged frame.
        SIG = SIG.reshape((-1, siz0[2])) 

        U,S,V = scipy.svd(SIG,full_matrices = False); # SVD decomposition
        SIG = U[:,n:N] @ S[n:N,n:N] @V[:,n:N].T; # high-pass filtering
        SIG = SIG.reshape(siz0)        
    else:
        raise ValueError("METHOD must be ''poly'', ''dct'', or ''svd''.")

    return SIG