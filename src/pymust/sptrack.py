from . import utils, smoothn
import numpy as np, scipy, scipy.sparse, scipy.sparse

def  sptrack(I :np.ndarray,param:utils.Param):
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


    I =  I.astype(float)

    #% Image size
    assert len(I.shape)==3,'I must be a 3-D array'
    M,N,P = I.shape

    #% Turn off warning messages for SMOOTHN
    #warn01 = warning('off','MATLAB:smoothn:MaxIter'); 
    #warn02 = warning('off','MATLAB:smoothn:SLowerBound');
    #warn03 = warning('off','MATLAB:smoothn:SUpperBound');

    if not utils.isfield(param,'winsize'):  #% Sizes of the interrogation windows
        raise ValueError('Window size(s) (PARAM.winsize) must be specified in the field PARAM.')

    assert param.get('winsize', None) is not None and param.winsize.shape[1]==2, 'PARAM.winsize must be a 2-column array.'
    tmp = np.diff(param.winsize, axis = 0)
    assert np.all(tmp<=0), 'The size of interrogation windows (PARAM.winsize) must decrease.'

    if not utils.isfield(param,'overlap'): # % Overlap
        param.overlap = 50

    assert utils.isscalar(param.overlap) and param.overlap>=0 and param.overlap<100, 'PARAM.overlap (in %) must be a scalar in [0,100[.'
    overlap = param.overlap/100

    if not utils.isfield(param,'ROI'): #% Region of interest
        ROI = np.all(np.isfinite(I),2)
    else:
        ROI = param.ROI
        assert utils.islogical(ROI) and ROI.shape == (M, N), 'PARAM.ROI must be a binary image the same size as I(:,:,1).'


    I[np.tile(np.logical_not(ROI), [1, 1, P])] = np.nan # NaNing outside the ROI

    if not utils.isfield(param,'iminc'): #% Step increment
        param.iminc = 1

    assert param.iminc>0 and param.iminc==np.round(param.iminc), 'PARAM.iminc must be a positive integer.'
    assert param.iminc<P, 'PARAM.iminc must be < size(I,3).'

    if not utils.isfield(param,'subpix'): #% Step increment
        param.subpix = 'PF'


    #%-- Spatial gradients (for the Optical Flow "OF" subpixel method)
    if param.subpix == 'OF':
        dIdj,dIdi = np.gradient(I)



    i = []
    j = []
    m = []
    n = []
    options = utils.Options()
    options.TolZ = .1;# % will be used in SMOOTHN

    for kk, wSize in enumerate(param.winsize):
        
        #% Window centers
        ic0 = (2*i+m-1)/2
        jc0 = (2*j+n-1)/2
        
        #% Size of the interrogation window
        m = wSize[0]
        n = wSize[1]
        
        #% Positions (row,column) of the windows (left upper corner)
        inci = int(np.ceil(m*(1-overlap)))
        incj = int(np.ceil(n*(1-overlap));)
        j,i = np.meshgrid( np.arange(0, N-n, incj),np.arange(0, N-m, inci));
        
        #% Size of the displacement-field matrix
        siz = np.floor([(M-m)/inci (N-n)/incj])+1
        
        #% Window centers
        ic = (2*i+m-1)/2
        jc = (2*j+n-1)/2
        
        if kk>1:
            #% Interpolation onto the new grid
            di = interp2(jc0,ic0,di,jc,ic,'*cubic')
            dj = interp2(jc0,ic0,dj,jc,ic,'*cubic')
            
            #% Extrapolation (remove NaNs)
            dj = rmnan(di+1j*dj,2)
            di = dj.real
            dj = dj.imag
            di = np.round(di)
            dj = np.round(dj)
        else:
            di = np.zeros(siz)
            dj = di

        
        #% Hanning window
        H = scipy.signal.windows.hanning(n).reshape((1,-1))*scipy.signal.windows.hanning(m).reshape((-1,1))
        
        
        C = np.zeros(siz)# % will contain the correlation coefficients
        
        for k, _ in enumerate(i):
            
            #%-- Split the images into small windows
            if i[k]+di[k]>=0 and j[k]+dj[k]>=0 and \
                    i[k]+di[k]+m-1<M and j[k]+dj[k]+n-1<N:
                I1w = I[ \
                    np.arange(i[k],i[k]+m),\
                    np.arange(j[k],j[k]+n),\
                    :-param.iminc,\
                    : ]
                I2w =I[ \
                    np.arange(i[k] + di[k],i[k]+di[k] + m),\
                    np.arange(j[k] + dj[k] ,j[k]+ dj[k] +n),\
                    param.iminc:,\
                    : ]
            else:
                di[k] =  np.nan
                dj[k] = np.nan
                continue
            
            if np.any(np.isnan(I1w+I2w)):
                di[k] = np.nan
                dj[k] = np.nan
                continue
            
            #%-- FFT-based cross-correlation
            R = np.fft.rfft2((I2w- np.mean(I2w))*H) * \
                np.conj(np.fft.rfft2((I1w-np.mean(I1w))*H))
            #%- 
            R2 = R/(np.abs(R)+utils.eps()) #% normalized x-corr
            R2 = np.mean(R2,2)
            R2 = np.fft.irfft2(R2)
            C[k] = np.max(R2)
            #% C = correlation coefficients: will be used as weights in SMOOTHN
            #%--
            R = np.sum(R,2); % ensemble correlation
            R = np.fft.irfft2(R); % x-correlation
            
            #%-- Peak detection + Pixelwise displacement
            #%- Circular shift (i.e. make the peak centered, same as FFTSHIFT):
            m1 = int(np.floor(m/2))
            n1 = int(np.floor(n/2))
            R = R([m-m1+1:m 1:m-m1],[n-n1+1:n 1:n-n1]);
            #% --
            #% [~,idx] = max(R,[],'all','linear');
            [~,idx] = max(R(:));
            di0 = mod(idx-1,m)+1; % line number
            dj0 = (idx-di0)/m+1; % column number
            
            %-- Total displacement (with pixelwise increment)
            di(k) = di(k) + di0-m1-1;
            dj(k) = dj(k) + dj0-n1-1;
            
            %--- Subpixel-motion correction ---%
            if strcmpi(param.subpix,'PF') || kk<size(param.winsize,1)
                %
                % Parabolic Peak Fitting)
                %
                if di0>1 && di0<m
                    di(k) = di(k) + (R(di0-1,dj0)-R(di0+1,dj0))/...
                        (2*R(di0-1,dj0)-4*R(di0,dj0)+2*R(di0+1,dj0));
                end
                if dj0>1 && dj0<n
                    dj(k) = dj(k) + (R(di0,dj0-1)-R(di0,dj0+1))/...
                        (2*R(di0,dj0-1)-4*R(di0,dj0)+2*R(di0,dj0+1));
                end
                
            elseif strcmpi(param.subpix,'OF')
                %
                % Optical flow (Lucas Kanade)
                %
                if i(k)+di(k)>0 && j(k)+dj(k)>0 &&...
                        i(k)+di(k)+m-2<M && j(k)+dj(k)+n-2<N
                    %--
                    I2w = I(i(k)+di(k):i(k)+di(k)+m-1,...
                        j(k)+dj(k):j(k)+dj(k)+n-1,1+param.iminc:end,:);
                    dIdtw = I2w-I1w;
                    dIdiw = dIdi(i(k)+di(k):i(k)+di(k)+m-1,...
                        j(k)+dj(k):j(k)+dj(k)+n-1,1+param.iminc:end,:);
                    dIdjw = dIdj(i(k)+di(k):i(k)+di(k)+m-1,...
                        j(k)+dj(k):j(k)+dj(k)+n-1,1+param.iminc:end,:);
                    %--
                    a = sum(dIdiw(:).^2); d = sum(dIdjw(:).^2);
                    b = sum(dIdiw(:).*dIdjw(:));
                    A = [a d;d b];
                    b = -[sum(dIdiw(:).*dIdtw(:)); sum(dIdjw(:).*dIdtw(:))];
                    % tmp = A\b;
                    tmp = lsqminnorm(A,b);
                    %--
                    di(k) = di(k) + tmp(1);
                    dj(k) = dj(k) + tmp(2);
                else % (parabolic peak fitting)
                    if di0>1 && di0<m
                        di(k) = di(k) + (R(di0-1,dj0)-R(di0+1,dj0))/...
                            (2*R(di0-1,dj0)-4*R(di0,dj0)+2*R(di0+1,dj0));
                    end
                    if dj0>1 && dj0<n
                        dj(k) = dj(k) + (R(di0,dj0-1)-R(di0,dj0+1))/...
                            (2*R(di0,dj0-1)-4*R(di0,dj0)+2*R(di0,dj0+1));
                    end
                end
            end

            
        end
        
        %-- Weighted robust smoothing
        if kk==size(param.winsize,1), options.TolZ = 1e-3; end
        di, dj = smoothn.smoothn((di,dj),np.sqrt(C),'robust',options)
        

    if utils.isfield(param,'ROI'):
        j,i = np.meshgrid(np.arange(N), np.arange(M))
        ROI = interp2(j,i,ROI,jc,ic,'*nearest')
        di[np.logical_not(ROI)] = np.nan
        dj[np.logical_not(ROI)] = np.nan
    end

    return di,dj



def rmnan(x,order):

    #% Remove NaNs by inter/extrapolation
    #% see also INPAINTN
    #% Written by Louis Le Tarnec, RUBIC, 2012

    sizx =x.shape
    W = np.logical_not(np.isinf(x))

    x[np.logical_not(W)] = 0
    W = W.flatten()
    x = x.flatten()

    missing_values,_ = np.where(W==0)

    #% Matrix defined by Buckley (equation 23)
    #% Biometrika (1994), 81, 2, pp. 247-58
    d = len(sizx)
    for i,n in enumerate(sizx):
        e = np.ones(n)
        e_center = -2*np.ones(n)
        e_center[0] = -1
        e_center[-1] = -1
        K = scipy.sparse.spdiags([e, e_center, e],[-1,0,1],n,n)
        M = 1
        for j in range(d):
            if j==i:
                M = scipy.sparse.kron(K,M)
            else:
                m = sizx[j]
                I = scipy.sparse.spdiags(e, 0 ,m,m)
                M = scipy.sparse.kron(I,M)
        if i==0:
            A = M
        else:
            A = A+M

    A = np.linalg.matrix_power(A,order)

    #% Linear system to be solved
    x2 = -A@x
    x2 = x2[missing_values]
    A = A[missing_values, missing_values]

    #% Solution
    x2 = np.linalg.linsolve(A, x2)
    x[missing_values] = x2
    y = x.reshape(sizx)
    return y
