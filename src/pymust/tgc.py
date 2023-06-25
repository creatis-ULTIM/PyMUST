import numpy as np,scipy, scipy.signal
import itertools
from . import utils
def tgc(S: np.ndarray):
    """
    %TGC Time-gain compensation for RF or IQ signals
    %   TGC(RF) or TGC(IQ) performs a time-gain compensation of the RF or IQ
    %   signals using a decreasing exponential law. Each column of the RF/IQ
    %   array must correspond to a single RF/IQ signal over (fast-) time.
    %
    %   [~,C] = TGC(RF) or [~,C] = TGC(IQ) also returns the coefficients used
    %   for time-gain compensation (i.e. new_SIGNAL = C.*old_SIGNAL)
    %
    %
    %   This function is part of MUST (Matlab UltraSound Toolbox).
    %   MUST (c) 2020 Damien Garcia, LGPL-3.0-or-later
    %
    %   See also RF2IQ, DAS.
    %
    %   -- Damien Garcia -- 2012/10, last update 2020/05
    %   website: <a
    %   href="matlab:web('https://www.biomecardio.com')">www.BiomeCardio.com</a>
    """

    siz0 = S.shape

    if not utils.iscomplex(S): #% we have RF signals
        C = np.mean(np.abs(scipy.signal.hilbert(S, axis = 0)),1)
        #% C = median(abs(hilbert(S)),2);
    else:  #% we have IQ signals
        C = np.mean(np.abs(S),1)
        # C = median(abs(S),2);
    n = len(C)
    n1 = int(np.ceil(n/10))
    n2 = int(np.floor(n*9/10))
    """
    % -- Robust linear fitting of log(C)
    % The intensity is assumed to decrease exponentially as distance increases.
    % A robust linear fitting is performed on log(C) to seek the TGC
    % exponential law.
    % --
    % See RLINFIT for details
    """
    N = 200# ; % a maximum of N points is used for the fitting
    p = min(N/(n2-n1)*100,100)
    slope,intercept = rlinfit(np.arange(n1,n2), np.log(C[n1:n2]),p)

    C = np.exp(intercept+slope*np.arange(n).reshape((-1,1)))
    C = C[0]/C
    S = S*C

    S = S.reshape(siz0)
    return S, C

def rlinfit(x,y,p):
    """
    %RLINFIT   Robust linear regression
    %   See the original RLINFIT function for details
    """
    N = len(x)
    I = np.random.permutation(N)
    n = int(np.round(N*p/100))
    I = I[:n]
    x = x[I]
    y = y[I]

    #Not sure it is the best option, what about some regression with regularisation?
    if True:
        C = np.array( [ (i,j) for i,j in  itertools.combinations(np.arange(n), 2)] )
    else:
        pass
    slope = np.median( (y[C[:,1]]-y[C[:,0]])  / (x[C[:,1]]-x[C[:,0]]) )
    intercept = np.median(y-slope*x)
    return slope, intercept
