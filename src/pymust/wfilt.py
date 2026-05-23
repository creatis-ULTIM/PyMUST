"""
.. note:: Documentation auto-generated with Claude (claude-sonnet-4-6).
"""
import numpy as np,scipy, logging
import typing

def wfilt(SIG: np.ndarray, method: str, n: int) -> np.ndarray:
    """Wall (clutter) filter for RF or I/Q Doppler signals.

    High-pass filters a 3-D array of RF or I/Q signals along the slow-time
    axis to suppress stationary tissue clutter.

    Parameters
    ----------
    SIG : np.ndarray
        3-D array of shape ``(fast_time, elements, slow_time)``.
        Each column (first dim) is one RF/IQ signal over fast-time.
        The third dimension is the slow-time (ensemble) axis.
    method : str
        Filtering method (case-insensitive). One of:

        - ``'poly'`` – Nth-degree polynomial regression subtraction using
          orthogonal Legendre polynomials. ``n >= 0``; ``n = 0`` removes
          the slow-time mean only.
        - ``'dct'``  – Truncated DCT: removes the ``n`` lowest slow-time
          frequency components. ``n >= 1``.
        - ``'svd'``  – Truncated SVD: removes the top ``n`` singular
          vectors (greatest singular values). ``n >= 1``.
    n : int
        Polynomial degree or number of low-frequency / singular-vector
        components to remove (meaning depends on *method*).

    Returns
    -------
    np.ndarray
        Clutter-filtered signal, same shape as *SIG*.

    Notes
    -----
    Translated from the MATLAB MUST toolbox (Damien Garcia, 2014–2023).

    .. warning::
        This Python port has not been fully tested.

    See Also
    --------
    iq2doppler, rf2iq
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
        
        assert n>0, 'N must be >0 with the "dct" method.'
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
        
        assert n>0,'N must be >0 with the "svd" method.'
        assert N>=n,'The packet length must be >=N.'
        
        #% Each column represents a column-rearranged frame.
        SIG = SIG.reshape((-1, siz0[2])) 

        U,S,V = scipy.svd(SIG,full_matrices = False); # SVD decomposition
        SIG = U[:,n:N] @ S[n:N,n:N] @V[:,n:N].T; # high-pass filtering
        SIG = SIG.reshape(siz0)        
    else:
        raise ValueError('METHOD must be "poly", "dct", or "svd".')

    return SIG