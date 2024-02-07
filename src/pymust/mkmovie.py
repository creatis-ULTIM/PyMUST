from . import utils, pfield
import numpy as np, scipy, scipy.signal, logging
import matplotlib.pyplot as plt, matplotlib
from matplotlib.animation import FuncAnimation, PillowWriter

def mkmovie(*varargin):
    """
    How to call:
    
    mkmovie(delaysTX,param)
    mkmovie(x, z,RC,delaysTX,param)

    You can add a path (in string format) as last argument to save the movie as a gif.
    You can also add an option structure as last argument (if there is no path argument) 
    or second-last (in case there is a path argument) that will give otpions for the computation.

    Returns
    F: 3D matrix with the pressure fields as a function of time (the last dimension encodes time)
    info: struct with information on the movie (fps and grid)
    param: same parameter as input, only kept for matlab compatibility. To remove in a future.

    %MKMOVIE   Create movie frames and animated GIF for wave propagation
    %   F, info, param = MKMOVIE(DELAYS,PARAM) simulates ultrasound RF radio-frequency
    %   signals by using PFIELD and returns movie frames. The array elements
    %   are excited at different time delays, given by DELAYS (in s). The
    %   characteristics of the transmit and receive must be given in the
    %   structure PARAM (see below for details).
    %
    %   Note: MKMOVIE works in a 2-D space.
    %
    %
    %   By default, the ROI is of size 2L-by-2L, with L being the aperture
    %   length of the array [i.e. L = pitch*(Number_of_elements-1)], and its
    %   resolution is 50 pix/cm. These values can be modified through
    %   PARAM.movie:
    %       e.g. PARAM.movie = [ROI_width, ROI_height, resolution]
    %       IMPORTANT: Pay attention to the units!
    %                  Width and height are in cm, resolution is in pix/cm.
    %
    %   The output variable F is a 8-bit 3-D array that contains the frames
    %   along the third dimension.
    %
    %   MKMOVIE uses PFIELD during transmit and receive. The parameters that
    %   must be included in the structure PARAM are similar as those in PFIELD.
    %
    %   PARAM is a structure that contains the following fields:
    %   -------------------------------------------------------
    %       *** MOVIE PROPERTIES ***
    %   0)  PARAM.movie = [width height resolution duration fps]
    %       IMPORTANT: Pay attention to the units!
    %            ROI width and height are in cm, resolution is in pix/cm,
    %            duration is in s, fps is frame per second.
    %       The default is [2L 2L 50 15 10], with L (in cm!) being the aperture
    %       length of the array [i.e. L = pitch*(Number_of_elements-1)]
    %
    %       *** TRANSDUCER PROPERTIES ***
    %   1)  PARAM.fc: central frequency (in Hz, REQUIRED)
    %   2)  PARAM.pitch: pitch of the linear array (in m, REQUIRED)
    %   3)  PARAM.width: element width (in m, REQUIRED)
    %       or PARAM.kerf: kerf width (in m, REQUIRED)
    %       note: kerf = pitch-width 
    %   4)  PARAM.radius: radius of curvature (in m)
    %            The default is Inf (rectilinear array)
    %   5)  PARAM.bandwidth: pulse-echo (2-way) 6dB fractional bandwidth (in %)
    %            The default is 75%.
    %   6)  PARAM.baffle: property of the baffle:
    %            'soft' (default), 'rigid' or a scalar >= 0.
    %            See "Note on BAFFLE property" in PFIELD for details
    %
    %       *** MEDIUM PARAMETERS ***
    %   7)  PARAM.c: longitudinal velocity (in m/s, default = 1540 m/s)
    %   8)  PARAM.attenuation: attenuation coefficient (dB/cm/MHz, default: 0)
    %            Notes: A linear frequency-dependence is assumed.
    %                   A typical value for soft tissues is ~0.5 dB/cm/MHz.
    %
    %       *** TRANSMIT PARAMETERS ***
    %   9)  PARAM.TXapodization: transmision apodization (default: no apodization)
    %   10) PARAM.TXnow: pulse length in number of wavelengths (default: 1)
    %            Use PARAM.TXnow = Inf for a mono-harmonic signal.
    %   11) PARAM.TXfreqsweep: frequency sweep for a linear chirp (default: [])
    %            To be used to simulate a linear TX chirp.
    %            See "Note on CHIRP signals" in PFIELD for details
    %
    %   Other syntaxes:
    %   --------------
    %   F = MKMOVIE(X,Z,RC,DELAYS,PARAM) also simulates backscattered echoes.
    %   The scatterers are characterized by their coordinates (X,Z) and
    %   reflection coefficients RC. X, Z and RC must be of same size.
    %
    %   [F,INFO] = MKMOVIE(...) returns image information in the structure
    %   INFO. INFO.Xgrid and INFO.Zgrid are the x- and z-coordinates of the
    %   image. INFO.TimeStep is the time step between two consecutive frames.
    %
    %   [F,INFO,PARAM] = MKMOVIE(...) updates the fields of the PARAM
    %   structure.
    %
    %   
    %   OPTIONS:
    %   -------
    %      %-- FREQUENCY SAMPLES --%
    %   1) Only frequency components of the transmitted signal in the range
    %      [0,2fc] with significant amplitude are considered. The default
    %      relative amplitude is -60 dB in MKMOVIE. You can change this value
    %      by using the following:
    %          [...] = MKMOVIE(...,OPTIONS),
    %      where OPTIONS.dBThresh is the threshold in dB (default = -60).
    %   ---
    %      %-- FULL-FREQUENCY DIRECTIVITY --%   
    %   2) By default, the directivity of the elements depends only on the
    %      center frequency. This makes the calculation faster. To make the
    %      directivities fully frequency-dependent, use: 
    %          [...] = MKMOVIE(...,OPTIONS),
    %      with OPTIONS.FullFrequencyDirectivity = true (default = false).
    %   ---
    %       %-- ELEMENT SPLITTING --%   
    %   3)  Each transducer element of the array is split into small segments.
    %       The length of these small segments must be small enough to ensure
    %       that the far-field model is accurate. By default, the elements are
    %       split into M segments, with M being defined by:
    %           M = ceil(element_width/smallest_wavelength);
    %       To modify the number M of subelements by splitting, you may adjust
    %       OPTIONS.ElementSplitting. For example, OPTIONS.ElementSplitting = 1
    %   ---
    %       %-- WAIT BAR --%   
    %   4)  If OPTIONS.WaitBar is true, a wait bar appears (only if the number
    %       of frequency samples >10). Default is true.
    %   ---
    %   
    %   CREATE an animated GIF:
    %   ----------------------
    %   [...] = MKMOVIE(...,FILENAME) creates a 10-fps animated GIF to the file
    %   specified by FILENAME. The duration of the animated GIF is ~15 seconds.
    %   You can modify the duration and fps by using PARAM.movie (see above).
    %
    %   Example #1: a diverging wave from a phased-array transducer
    %   ----------
    %   % Phased-array @ 2.7 MHz:
    %   param = getparam('P4-2v');
    %   % TX time delays for a 90-degree wide diverging wave:
    %   dels = txdelay(param,0,pi/2);
    %   % Scatterers' position:
    %   n = 20;
    %   x = rand(n,1)*8e-2-4e-2;
    %   z = rand(n,1)*10e-2;
    %   % Backscattering coefficient
    %   RC = (rand(n,1)+1)/2;
    %   % Image size (in cm)
    %   param.movie = [8 10];
    %   % Movie frames
    %   [F,info] = mkmovie(x,z,RC,dels,param);
    %   % Check the movie frames
    %   figure
    %   colormap([1-hot(128); hot(128)]);
    %   for k = 1:size(F,3)
    %       image(info.Xgrid,info.Zgrid,F(:,:,k))
    %       hold on
    %       scatter(x,z,5,'w','filled')
    %       hold off
    %       axis equal off
    %       title([int2str(info.TimeStep*k*1e6) ' \mus'])
    %       drawnow
    %   end
    %
    %   Example #2: an animated GIF of a focused wave
    %   ----------
    %   % Phased-array @ 2.7 MHz
    %   param = getparam('P4-2v');
    %   % Focus location at xf = 0 cm, zf = 3 cm
    %   xf = 0; zf = 3e-2; % focus position (in m)
    %   % Transmit time delays (in s) 
    %   txdel = txdelay(xf,zf,param); % in s
    %   % Define the image size (in cm) and its resolution (in pix/cm)
    %   param.movie = [3 6 100];
    %   % Create an animated GIF
    %   mkmovie(txdel,param,'focused_wave.gif');
    %   % Open the GIF in your browser
    %   web('focused_wave.gif')
    %
    %   This function is part of <a
    %   href="matlab:web('https://www.biomecardio.com/MUST')">MUST</a> (Matlab UltraSound Toolbox).
    %   MUST (c) 2020 Damien Garcia, LGPL-3.0-or-later
    %
    %   See also PFIELD, SIMUS, TXDELAY.
    %
    %   -- Damien Garcia -- 2017/10, last update 2023/02/01
    %   website: <a
    %   href="matlab:web('http://www.biomecardio.com')">www.BiomeCardio.com</a>
    """
    nargin = len(varargin)
    assert 2 <= nargin <= 7, "Wrong number of input arguments."

    if isinstance(varargin[-1],str):
        gifname = varargin[-1];
        isGIF = True;
        Nargin = nargin-1;
    else:
        isGIF = False;
        Nargin = nargin;

    if Nargin == 2:# %#mkmovie(delaysTX,param)
            delaysTX = varargin[0]
            param = varargin[1]
            x = []; 
            z = []; 
            RC = [];
            options = utils.Options()
    elif Nargin == 3: #% mkmovie(delaysTX,param,options)
            delaysTX = varargin[0]
            param = varargin[1]
            options = varargin[2]
            x = []; 
            z = []; 
            RC = []; 
    elif Nargin == 5: #mkmovie(x,z,RC,delaysTX,param)
            x = varargin[0]
            z = varargin[1]
            RC = varargin[2]
            delaysTX = varargin[3]
            param = varargin[4]
            options = utils.Options()
    elif Nargin == 6: #% mkmovie(x,z,RC,delaysTX,param,options)
            x = varargin[0]
            z = varargin[1]
            RC = varargin[2]
            delaysTX = varargin[3]
            param = varargin[4]
            options = varargin[5]
    else:
        raise ValueError('Wrong input arguments')

    param = param.ignoreCaseInFieldNames()
    options = options.ignoreCaseInFieldNames()
    options.CallFun = 'mkmovie'


    #%------------------------%
    #% CHECK THE INPUT SYNTAX %
    #%------------------------%

    assert isinstance(param, utils.Param),'The structure PARAM is required.'
    if not utils.isEmpty(x) or   not utils.isEmpty(z)  or not utils.isEmpty(RC):
        assert np.array_equal(x.shape, z.shape) and np.array_equal(x.shape, RC.shape),'X, Z and RC must be of same size.'

    #%-- Check if syntax errors may appear when using PFIELD
    opt = options.copy()
    pfield(None, None, None, delaysTX,param,opt)

    #%-- Movie properties
    NumberOfElements = param.Nelements; # number of array elements
    L = param.pitch*(NumberOfElements-1)
    if not 'movie' in param:
        # NOTE: width and height are in cm
        param.movie = np.array([200*L, 200*L, 50, 15, 10]) # default
    else:
        assert utils.isnumeric(param.movie) and len(param.movie)>1 and len(param.movie)<6 and np.all(param.movie>0), 'PARAM.movie must contain two to five positive parameters.'


    #% default resolution = 50 pix/cm, default duration = 15 s, default fps = 10
    paramMOVdefault = np.array([np.nan,np.nan,50, 15, 10])
    n = len(param.movie)
    param.movie = np.concatenate([param.movie,  paramMOVdefault[n:]] )


    #%-- dB threshold (i.e. faster computation if lower)
    if not utils.isfield(options,'dBThresh') or not utils.isEmpty(options.dBThresh):
        options.dBThresh = -60 # default is -60dB in MKMOVIE
    assert np.isscalar(options.dBThresh) and utils.isnumeric(options.dBThresh) and options.dBThresh<0,'OPTIONS.dBThresh must be a negative scalar.'

    #-- Frequency step (scaling factor)
    # The frequency step is determined automatically. It is tuned to avoid
    # aliasing in the temporal domain. The frequency step can be adjusted by
    # using a scaling factor. For a smoother result, you may use a scaling
    # factor<1.
    if not utils.isfield(options,'FrequencyStep'):
        options.FrequencyStep = 1
    assert np.isscalar(options.FrequencyStep) and \
        utils.isnumeric(options.FrequencyStep) and options.FrequencyStep>0, \
        'OPTIONS.FrequencyStep must be a positive scalar.'
    
    if options.FrequencyStep>1:
        logging.warning('MUST:FrequencyStep \n OPTIONS.FrequencyStep is >1: aliasing may be present!')


    #%-------------------------------%
    #% end of CHECK THE INPUT SYNTAX %
    #%-------------------------------%


    #%-- Image grid
    ROIwidth = param.movie[0]*1e-2;  # image width (in m)
    ROIheight = param.movie[1]*1e-2; # image height (in m)
    pixsize = 1/param.movie[2]*1e-2; #% pixel size (in m)
    xi = np.arange(pixsize/2,ROIwidth  + pixsize/2, pixsize);
    zi = np.arange(pixsize/2,ROIheight + pixsize/2, pixsize);
    xi, zi = np.meshgrid(xi-np.mean(xi), zi);


    #%-- Frequency sampling
    maxD = np.hypot((ROIwidth+L)/2,ROIheight); # maximum travel distance
    df = 1/2/(maxD/param.c);
    df = df*options.FrequencyStep;
    Nf = int(2*np.ceil(param.fc/df)+1); #% number of frequency samples


    #%-- Run PFIELD to calculate the RF spectra
    SPECT = np.zeros([Nf,np.prod(xi.shape)], dtype = np.complex64); # will contain the RF spectra
    options.FrequencyStep = df;
    options.ElementSplitting = 1;
    options.RC = RC;
    options.x = x;
    options.z = z;
    #%-
    #% we need idx...
    opt = options.copy();
    opt.x = [];
    opt.z = []; 
    opt.RC = [];
    opt.computeIndices = True
    _,_,idx = pfield([],[], [], delaysTX,param,opt);
    #-
    _, SPECT[idx,:], _ = pfield(xi, [], zi,delaysTX,param,options);

    #-- IFFT to recover the time-resolved signals
    #%
    #F = SPECT #; clear SPECT
    F = SPECT
    F = F.reshape([Nf, xi.shape[0], xi.shape[1]]) #reshape(F,Nf,size(xi,1),size(xi,2));
    F = utils.shiftdim(F,1)

    #F = np.concatenate([F,
    #                    np.conj(np.flip(F[:,:,2:-1],3))
    #                            ], 
    #                    axis = 2
    #                    );
    

    #%
    F = np.fft.irfft(F, axis = 2); # Note GB: not Sure if you need the expanded version covering positive and negative, or the single spectra is enoguh
    #%

    #%
    F = np.flip(F,2)
    F = F[:,:,:int(np.round(F.shape[2]//2))];
    F = F/np.max(np.abs(F))

    
    if utils.isfield(param,'gamma') and param.gamma!=1: # % gamma compression
        F = np.power(np.abs(F), param.gamma*np.sign(F))
    F = ((F+1)/2*255).astype(np.uint8);
    #%

    #%-- some information about the movie
    info = utils.dotdict()
    info.Xgrid = xi[0, :] # % in m
    info.Zgrid = zi[:, 0] # % in m
    info.TimeStep = maxD/param.c/F.shape[2] #% in s

    #%-- animated GIF
    if isGIF:
        plotScatters = options.get('plotScatterers', False)
        # Cretae the colormap, see https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html
        N = 256
        vals = np.ones((N, 4))
        vals[128:, :] = [plt.cm.hot(0+i * 2) for i in range(128)  ]
        vals[:128, :] = [plt.cm.hot(1 + i * 2) for i in range(128)  ]
        vals[:128, :3] = 1 - vals[:128, :3]
        newcmp = matplotlib.colors.ListedColormap(vals)
        #%- the matrix f contains the scatterers
        f = np.zeros(F.shape[:2])
        if not utils.isEmpty(x):
            maxRC = 1; # max(RC(:));
            for k,_ in enumerate(x):
                i = np.argmin(np.abs(zi[:,0]-z[k]))
                j = np.argmin(np.abs(xi[:,0]-x[k]))
                f[i,j] = RC[k]/maxRC

            n = int(2*np.round(param.movie[2]/10)+1)
            window1d = np.abs(np.blackman(n))
            window2d = np.sqrt(np.outer(window1d,window1d))
            f = scipy.signal.convolve2d(f,window2d,'same')

        f = (f*128).astype(np.uint8)
        
        # NOTE GB: put it later somehow
        #%- add the signature
        #% Please do not remove it
        #if f.shape>37 && size(f,2)>147
        #    f(end-36:end-5,6:147) = Signature/2;
        #else
        #    f = 0;
        #end
        
        #map = np.flipud([1-hot(128); hot(128)]);
            
        #%- create the GIF movie

        # Define the colormap
        vals =  np.array([matplotlib.cm.hot(2*i) for i in range(127)] + [matplotlib.cm.hot(2*i) for i in range(128)])
        vals[:127, :3] = 1 - vals[:127, :3]
        cm2 = matplotlib.colors.LinearSegmentedColormap.from_list('hot2',vals)



        Tmov = param.movie[3]# % movie duration in s
        fps = param.movie[4] # % frame per second
        
        nk = int(np.round(F.shape[2]/(Tmov*fps))) #; % increment
        ks = np.arange(0,F.shape[2],nk)

        interactive = matplotlib.is_interactive()
        if interactive:
            plt.ioff()

        fig,ax = plt.subplots()

        def animate(i):
            ax.clear()
            im = ax.imshow(F[:, :,ks[i]], cmap = cm2, extent=[info.Xgrid[0]*1e2,info.Xgrid[-1]*1e2,info.Zgrid[-1]*1e2,info.Zgrid[0]*1e2])
            if plotScatters:
                dx = info.Xgrid[1] - info.Xgrid[0]
                dz = info.Zgrid[1] - info.Zgrid[0]
                ax.scatter(x/dx,z/dz, s = 5, c = 'w', marker = 'o', facecolors = 'none')

            im.set_clim(0,255)
            plt.xlabel('x (cm)')
            plt.ylabel('z (cm)')
            return im, 
                
        ani = FuncAnimation(fig, animate, blit=True, repeat=False, frames=len(ks))    
        ani.save(gifname, dpi=300, writer=PillowWriter(fps=fps))
        plt.close(fig)
        if interactive:
            plt.ion()

    return F,info,param