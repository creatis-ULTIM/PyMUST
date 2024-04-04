import logging
from . import utils, smoothn
import numpy as np, scipy


def sptrack(I : np.ndarray, param : utils.Param) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    %SPTRACK   Speckle tracking using Fourier-based cross-correlation
    %   [Di,Dj] = SPTRACK(I,PARAM) returns the motion field [Di,Dj] that occurs
    %   from frame#k I(:,:,k) to frame#(k+1) I(:,:,k+1).
    %
    %   I must be a 3-D array, with I(:,:,k) corresponding to image #k. I can
    %   contain more than two images (i.e. size(I,3)>2). In such a case, an
    %   ensemble correlation is used.
    %
    %   >--- Try it: enter "sptrack" in the command window for an example ---<
    %
    %   Di,Dj are the displacements (unit = pix) in the IMAGE coordinate system
    %   (i.e. the "matrix" axes mode). The i-axis is vertical, with values
    %   increasing from top to bottom. The j-axis is horizontal with values
    %   increasing from left to right. The coordinate (1,1) corresponds to the
    %   center of the upper left pixel.
    %   To display the displacement field, you may use: quiver(Dj,Di), axis ij
    %
    %   PARAM is a structure that contains the parameter values required for
    %   speckle tracking (see below for details).
    %
    %   [Di,Dj,id,jd] = SPTRACK(...) also returns the coordinates of the points
    %   where the components of the displacement field are estimated.
    %
    %
    %   PARAM is a structure that contains the following fields:
    %   -------------------------------------------------------
    %   1) PARAM.winsize: Size of the interrogation windows (REQUIRED)
    %           PARAM.winsize must be a 2-COLUMN array. If PARAM.winsize
    %           contains several rows, a multi-grid, multiple-pass interro-
    %           gation process is used.
    %           Examples: a) If PARAM.winsize = [64 32], then a 64-by-32
    %                        (64 lines, 32 columns) interrogation window is used.
    %                     b) If PARAM.winsize = [64 64;32 32;16 16], a 64-by-64
    %                        interrogation window is first used. Then a 32-by-32
    %                        window, and finally a 16-by-16 window are used.
    %   2) PARAM.overlap: Overlap between the interrogation windows
    %                     (in %, default = 50)
    %   3) PARAM.iminc: Image increment (for ensemble correlation, default = 1)
    %            The image #k is compared with image #(k+PARAM.iminc):
    %            I(:,:,k) is compared with I(:,:,k+PARAM.iminc)
    %   5) PARAM.ROI: 2-D region of interest (default = the whole image).
    %            PARAM.ROI must be a logical 2-D array with a size of
    %            [size(I,1),size(I,2)]. The default is all(isfinite(I),3).
    %
    %   NOTES:
    %   -----
    %   The displacement field is returned in PIXELS. Perform an appropriate
    %   calibration to get physical units.
    %
    %   SPTRACK is based on a multi-step cross-correlation method. The SMOOTHN
    %   function (see Reference below) is used at each iterative step for the
    %   validation and post-processing.
    %
    %
    %   Example:
    %   -------
    %   I1 = conv2(rand(500,500),ones(10,10),'same'); % create a 500x500 image
    %   I2 = imrotate(I1,-3,'bicubic','crop'); % clockwise rotation
    %   param.winsize = [64 64;32 32];
    %   [di,dj] = sptrack(cat(3,I1,I2),param);
    %   quiver(dj(1:2:end,1:2:end),di(1:2:end,1:2:end))
    %   axis equal ij
    %
    %
    %   References for speckle tracking 
    %   -------------------------------
    %   1) Garcia D, Lantelme P, Saloux É. Introduction to speckle tracking in
    %   cardiac ultrasound imaging. Handbook of speckle filtering and tracking
    %   in cardiovascular ultrasound imaging and video. Institution of
    %   Engineering and Technology. 2018.
    %   <a
    %   href="matlab:web('https://www.biomecardio.com/publis/eti18.pdf')">PDF download</a>
    %   2) Perrot V, Garcia D. Back to basics in ultrasound velocimetry:
    %   tracking speckles by using a standard PIV algorithm. IEEE International
    %   Ultrasonics Symposium (IUS). 2018
    %   <a
    %   href="matlab:web('https://www.biomecardio.com/publis/ius18.pdf')">PDF download</a>
    %   3) 	Joos P, Porée J, ..., Garcia D. High-frame-rate speckle tracking
    %   echocardiography. IEEE Trans Ultrason Ferroelectr Freq Control. 2018.
    %   <a
    %   href="matlab:web('https://www.biomecardio.com/publis/ieeeuffc18.pdf')">PDF download</a>
    %
    %
    %   References for smoothing 
    %   -------------------------------
    %   1) Garcia D, Robust smoothing of gridded data in one and higher
    %   dimensions with missing values. Computational Statistics & Data
    %   Analysis, 2010.
    %   <a
    %   href="matlab:web('http://www.biomecardio.com/pageshtm/publi/csda10.pdf')">PDF download</a>
    %   2) Garcia D, A fast all-in-one method for automated post-processing of
    %   PIV data. Experiments in Fluids, 2011.
    %   <a
    %   href="matlab:web('http://www.biomecardio.com/pageshtm/publi/expfluids10.pdf')">PDF download</a>
    %   
    %
    %   This function is part of MUST (Matlab UltraSound Toolbox).
    %   MUST (c) 2020 Damien Garcia, LGPL-3.0-or-later
    %
    %   See also SMOOTHN
    %
    %   -- Damien Garcia & Vincent Perrot -- 2013/02, last update: 2021/06/28
    %   website: <a
    %   href="matlab:web('http://www.biomecardio.com')">www.BiomeCardio.com</a>
    """


    #%------------------------%
    #% CHECK THE INPUT SYNTAX %
    #%------------------------%

    I = I.astype(float)

    # Image size
    assert len(I.shape) ==3,'I must be a 3-D array'
    M,N, P = I.shape

    if not utils.isfield(param, 'winsize'): # Sizes of the interrogation windows
        raise ValueError('Window size(s) (PARAM.winsize) must be specified in the field PARAM.')
    if isinstance(param.winsize, list):
        param.winsize = np.array(param.winsize)
    if isinstance(param.winsize, np.ndarray) and len(param.winsize.shape) == 1:
        param.winsize = param.winsize.reshape((1, -1), order = 'F')

    assert isinstance(param.winsize, np.ndarray) and len(param.winsize.shape) == 2 and  param.winsize.shape[1] == 2 and 'PARAM.winsize must be a 2-column array.'
    tmp = np.diff(param.winsize,1,0)
    assert np.all(tmp<=0), 'The size of interrogation windows (PARAM.winsize) must decrease.'

    if not utils.isfield(param,'overlap'): # Overlap
        param.overlap = 50;

    assert np.isscalar(param.overlap) and param.overlap>=0 and param.overlap<100, 'PARAM.overlap (in %) must be a scalar in [0,100[.'
    overlap = param.overlap/100;

    if not utils.isfield(param,'ROI'): # Region of interest
        param.ROI = np.all(np.isfinite(I),2)
    ROI = param.ROI;
    assert isinstance(ROI, np.ndarray) and ROI.dtype == bool and np.allclose(ROI.shape,[M, N]),  'PARAM.ROI must be a binary image the same size as I[]:,:,0].'
    

    I[np.tile(np.logical_not(ROI)[..., None],[1,1, P])] = np.nan; # NaNing outside the ROI

    if not utils.isfield(param,'iminc'): # Step increment
        param.iminc = 1

    assert param.iminc>0 and isinstance(param.iminc, int), 'PARAM.iminc must be a positive integer.'
    assert param.iminc<P, 'PARAM.iminc must be < I.shape[2].'

    if not utils.isfield(param,'subpix'): # Step increment
        param.subpix = 'PF'


    #%-- Spatial gradients (for the Optical Flow "OF" subpixel method)
    if param.subpix == 'OF':
        dIdj,dIdi = np.gradient(I)


    # Dummy initialisatino
    i = []; 
    j = [];
    m = []; 
    n = [];
    TolZ = .1; # will be used in SMOOTHN

    for kk  in range(param.winsize.shape[0]):
        
        # Window centers
        if kk >= 1:
            ic0 = (2*i+m)/2;
            ic0_array = (2*i0_array+m)/2
            jc0 = (2*j+n)/2;
            jc0_array = (2*j0_array+m)/2
            m0 = len(ic0_array)
            n0 = len(jc0_array)

        # Size of the interrogation window
        m = param.winsize[kk,0];
        n = param.winsize[kk,1];
        
        # Positions (row,column) of the windows (left upper corner)
        inci = np.ceil(m*(1-overlap)); incj = np.ceil(n*(1-overlap));
        i0_array= np.arange(0,M-m+1,inci, dtype=int)
        j0_array =np.arange(0,N-n+1,incj, dtype=int)
        j,i = np.meshgrid(j0_array,i0_array);
        # Size of the displacement-field matrix
        siz = (np.floor([(M-m)/inci, (N-n)/incj])+1).astype(int) # j.shape
        j = j.flatten(order = 'F'); i = i.flatten(order = 'F');   
        
        # Window centers
        ic = (2*i+m)/2;
        ic_array = (2*i0_array+m)/2
        jc = (2*j+n)/2;
        jc_array = (2*j0_array+n)/2
        
        if kk>=1:

            #% Interpolation onto the new grid
            di  = scipy.interpolate.interpn((jc0_array,ic0_array),di.reshape((m0,n0), order = 'F'),(jc,ic),'cubic', bounds_error = False, fill_value = np.nan)
            dj  = scipy.interpolate.interpn((jc0_array,ic0_array),dj.reshape((m0,n0), order = 'F'),(jc,ic),'cubic', bounds_error = False, fill_value = np.nan)

            #% Extrapolation (remove NaNs)
            dj = rmnan(di+1j*dj,2)
            di = dj.real; 
            dj = dj.imag;
            di = np.round(di); dj = np.round(dj);
        else:
            di = np.zeros(siz).flatten(order = 'F');
            dj = di.copy();
        
        
        #% Hanning window
        H = np.outer(scipy.signal.windows.hann(n+2)[1:-1], scipy.signal.windows.hann(m+2)[1:-1])
        
        
        C = np.zeros(siz).flatten(order = 'F'); # will contain the correlation coefficients
        
        for k,_ in enumerate(i): 
            #-- Split the images into small windows
            if i[k]+di[k]>=0 and j[k]+dj[k]>=0 and \
                    i[k]+di[k]+m<M  and j[k]+dj[k]+n<N:
                
                I1w = I[i[k]:i[k]+m,
                        j[k]:j[k]+n,
                        : -param.iminc]
                I2w = I[i[k] + int(di[k]):i[k] + int(di[k])+m,
                        j[k] + int(dj[k]):j[k] + int(dj[k])+n,
                        param.iminc:] # Why a new dimension?
            else:
                di[k] = np.nan; 
                dj[k] = np.nan;
                continue
            
            
            if np.any(np.isnan(I1w+I2w)):
                di[k] = np.nan; 
                dj[k] = np.nan;
                continue
            
            #%-- FFT-based cross-correlation
            # Note GB: could use real fft, but it s a bit of a mess to change everything
            R = scipy.fft.fft2((I2w-np.mean(I2w)) *H[..., None], axes = (0, 1)) *  \
                np.conj(scipy.fft.fft2((I1w-np.mean(I1w))*H[..., None], axes = (0, 1))) #GB: Recheck, maximum intensity, or absolute?

            #- 
            R2 = R/(np.abs(R)+1e-6); # normalized x-corr
            R2 = np.mean(R2, 2)
            R2 = scipy.fft.ifft2(R2, axes = (0, 1)).real
            C[k] = np.max(R2)
            # C = correlation coefficients: will be used as weights in SMOOTHN
            #--
            R = np.sum(R,2); # ensemble correlation
            R = scipy.fft.ifft2(R, axes = (0, 1)).real; # x-correlation

            #-- Peak detection + Pixelwise displacement
            #- Circular shift (i.e. make the peak centered, same as FFTSHIFT):
            m1 = int(np.floor(m/2)); 
            n1 = int(np.floor(n/2));
            #R = R[[m-m1:m, :m-m1],[n-n1:n, :n-n1]]
            R = scipy.fft.fftshift(R, axes = (0, 1))

            # --
            # [~,idx] = max(R,[],'all','linear');
            idx = np.argmax(R.T.flatten(order = "F")) # Make it consistent with matlab code, can just clean the code
            di0 = np.mod(idx,m); # line number
            dj0 = (idx-di0)//m; # column number
            #-- Total displacement (with pixelwise increment)
            di[k] = di[k] + di0-m1
            dj[k] = dj[k] + dj0-n1

            #%--- Subpixel-motion correction ---%
            if param.subpix == 'PF' or kk< param.winsize.shape[0]:
                #%
                #% Parabolic Peak Fitting)
                #%
                if di0>0 and di0<m -1:
                    di[k] = di[k] + (R[di0-1,dj0]-R[di0+1,dj0]) / \
                        (2*R[di0-1,dj0]-4*R[di0,dj0]+2*R[di0+1,dj0]);
                
                if dj0>0 and dj0<n - 1:
                    dj[k] = dj[k] + (R[di0,dj0-1]-R[di0,dj0+1]) / \
                        (2*R[di0,dj0-1]-4*R[di0,dj0]+2*R[di0,dj0+1]);

                
            elif param.subpix == 'OF':
                #%
                #% Optical flow (Lucas Kanade)
                #%
                if i[k]+di[k]>0 and j[k]+dj[k]>0 and \
                        i[k]+di[k]+m-2<M and j[k]+dj[k]+n-2<N:
                    #%--
                    I2w = I[i[k]+di[k]:i[k]+di[k]+m-1, \
                        j[k]+dj[k]:j[k]+dj[k]+n-1,1+param.iminc:,:]
                    dIdtw = I2w-I1w;
                    dIdiw = dIdi[i[k]+di[k]:i[k]+di[k]+m-1, \
                        j[k]+dj[k]:j[k]+dj[k]+n-1,1+param.iminc,:];
                    dIdjw = dIdj[i[k]+di[k]:i[k]+di[k]+m-1, \
                        j[k]+dj[k]:j[k]+dj[k]+n-1,1+param.iminc:,:];
                    #%--
                    a = np.sum(dIdiw**2); 
                    d = np.sum(dIdjw**2);
                    b = np.sum(dIdiw*dIdjw);
                    A = np.array([[a, d],[d, b]])
                    b = -np.array([np.sum(dIdiw*dIdtw),  np.sum(dIdjw*dIdtw)])
                    #% tmp = A\b;
                    tmp = scipy.linalg.lstsq(A,b);
                    #%--
                    di[k] = di[k] + tmp[0];
                    dj[k] = dj[k] + tmp[1];
                else: #% (parabolic peak fitting)
                    if di0>0 and di0<m -1:
                        di[k] = di[k] + (R[di0-1,dj0]-R[di0+1,dj0])/\
                            (2*R[di0-1,dj0]-4*R[di0,dj0]+2*R[di0+1,dj0]);
                    
                    if dj0>1 and dj0<n:
                        dj[k] = dj[k] + (R[di0,dj0-1]-R[di0,dj0+1])/\
                            (2*R[di0,dj0-1]-4*R[di0,dj0]+2*R[di0,dj0+1]);


        #-- Weighted robust smoothing
        if kk== param.winsize.shape[0] -1:
             TolZ = 1e-3; 
        print(np.mean(C))
        print('dj smooth =', np.nanstd(dj[dj ==dj]))
        print('di smooth =', np.nanstd(di[di ==di]))

        dj, _, _ = smoothn([di.reshape(siz, order = 'F'),
                            dj.reshape(siz, order = 'F')],
                            np.sqrt(C.reshape(siz, order = 'F')), isrobust = True,TolZ = TolZ);
        di = dj[0].flatten(order = 'F');
        dj = dj[1].flatten(order = 'F');
        print('dj smoothed =', np.nanstd(dj))
        print('di smoothed =', np.nanstd(di))


    if utils.isfield(param,'ROI'):
        #j,i = np.meshgrid(np.arange(N), np.arange(M));
        ROI = scipy.interpolate.interpn((np.arange(N), np.arange(M)),ROI,(jc,ic),method ='nearest');
        di[np.logical_not(ROI)] = np.nan;
        dj[np.logical_not(ROI)] = np.nan;
    return di.reshape(siz, order = 'F'), dj.reshape(siz, order = 'F'), ic, jc

def rmnan(x,order):
    # Remove NaNs by inter/extrapolation
    # see also INPAINTN
    # Written by Louis Le Tarnec, RUBIC, 2012

    sizx = x.shape;
    W = np.isfinite(x);

    x[np.logical_not(W)] = 0;
    W = W.flatten(order = 'F');
    x = x.flatten(order = 'F');

    missing_values = W==0

    #% Matrix defined by Buckley (equation 23)
    #% Biometrika (1994), 81, 2, pp. 247-58
    d = len(sizx);
    for i in range(d):
        n = sizx[i];
        e = np.ones(n, dtype = x.dtype);
        K = scipy.sparse.spdiags([e, -2*e, e],[-1,0,1], n, n).tocsr()
        K[0,0] = -1;
        K[n-1,n-1] = -1; ##ok
        M = 1;
        for j in range(d):
            if j==i:
                M = scipy.sparse.kron(K,M); 
            if j!=i:
                m = sizx[j];
                I = scipy.sparse.identity(m, dtype = x.dtype)
                M = scipy.sparse.kron(I,M);
        if i==0:
            A = M; 
        else:
            A = A+M; 
    
    A = A**order;
    

    #% Linear system to be solved
    x2 = -A@x;
    x2 = x2[missing_values];
    A = A.tocsr()[missing_values, :][:, missing_values];

    #% Solution
    x2 = scipy.sparse.linalg.spsolve(A,x2);
    x[missing_values] = x2;
    y = x.reshape(sizx, order = 'F');

    return y