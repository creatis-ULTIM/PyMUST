import numpy as np
from . import utils
def bmode(IQ : np.ndarray, DR: float= 40):

    """
    %BMODE   B-mode image from I/Q signals
    %   BMODE(IQ,DR) converts the I/Q signals (in IQ) to 8-bit log-compressed
    %   ultrasound images with a dynamic range DR (in dB). IQ is a complex
    %   whose real (imaginary) part contains the inphase (quadrature)
    %   component.
    %
    %   BMODE(IQ) uses DR = 40 dB;
    %
    %
    %   Example:
    %   -------
    %   #-- Analyze undersampled RF signals and generate B-mode images
    %   # Download experimental data (128-element linear array + rotating disk) 
    %   load('PWI_disk.mat')
    %   # Demodulate the RF signals with RF2IQ.
    %   IQ = rf2iq(RF,param);
    %   % Create a 2.5-cm-by-2.5-cm image grid.
    %   dx = 1e-4; % grid x-step (in m)
    %   dz = 1e-4; % grid z-step (in m)
    %   [x,z] = meshgrid(-1.25e-2:dx:1.25e-2,1e-2:dz:3.5e-2);
    %   % Create a Delay-And-Sum DAS matri  x with DASMTX.
    %   param.fnumber = []; % an f-number will be determined by DASMTX
    %   M = dasmtx(1i*size(IQ),x,z,param,'nearest');
    %   % Beamform the I/Q signals.
    %   IQb = M*reshape(IQ,[],32);
    %   IQb = reshape(IQb,[size(x) 32]);
    %   % Create the B-mode images with a -30dB range.
    %   I = bmode(IQb,30);
    %   % Display four B-mode images.
    %   for k = 1:4
    %   subplot(2,2,k)
    %   imshow(I(:,:,10*k-9))
    %   axis off
    %   title(['frame #' int2str(10*k-9)])
    %   end
    %   colormap gray
    %
    %
    %   This function is part of MUST (Matlab UltraSound Toolbox).
    %   MUST (c) 2020 Damien Garcia, LGPL-3.0-or-later
    %
    %   See also RF2IQ, TGC, SPTRACK.
    %
    %   -- Damien Garcia -- 2020/06
    %   website: <a
    %   href="matlab:web('https://www.biomecardio.com')">www.BiomeCardio.com</a>
    """
    assert utils.iscomplex(IQ),'IQ must be a complex array'

    I = np.abs(IQ) # real envelope

    if (DR >= 1):
        I = 20*np.log10(I/np.max(I))+DR
        I = (255*I/DR) #.astype(np.uint8) # 8-bit log-compressed image
    else:    
        I = np.power(I / np.max(I), DR)
        I *= 255

    I[I<0] = 0
    I[I>255] = 255

    return I.astype(np.uint8)