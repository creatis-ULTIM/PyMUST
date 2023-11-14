import numpy as np
from . import utils


def getparam(probe : str): 
    #GETPARAM   Get parameters of a uniform linear or convex array
#   PARAM = GETPARAM opens a dialog box which allows you to select a
#   transducer whose parameters are returned in PARAM.
    
    #   PARAM = GETPARAM(PROBE), where PROBE is a string, returns the prameters
#   of the transducer given by PROBE.
    
    #   The structure PARAM is used in several functions of MUST (Matlab
#   UltraSound Toolbox). The structure returned by GETPARAM contains only
#   the fields that describe a transducer. Other fields may be required in
#   some MUST functions.
    
    #   PROBE can be one of the following:
#   ---------------------------------
#     1) 'L11-5v' (128-element, 7.6-MHz linear array)
#     2) 'L12-3v' (192-element, 7.5-MHz linear array)
#     3) 'C5-2v' (128-element, 3.6-MHz convex array)
#     4) 'P4-2v' (64-element, 2.7-MHz phased array)
    
    #   These are the <a
#   href="matlab:web('https://verasonics.com/verasonics-transducers/')">Verasonics' transducers</a>.
#   Feel free to complete this list for your own use.
    
    #   PARAM is a structure that contains the following fields:
#   --------------------------------------------------------
#   1) PARAM.Nelements: number of elements in the transducer array
#   2) PARAM.fc: center frequency (in Hz)
#   3) PARAM.pitch: element pitch (in m)
#   4) PARAM.width: element width (in m)
#   5) PARAM.kerf: kerf width (in m)
#   6) PARAM.bandwidth: 6-dB fractional bandwidth (in #)
#   7) PARAM.radius: radius of curvature (in m, Inf for a linear array)
#   8) PARAM.focus: elevation focus (in m)
#   9) PARAM.height: element height (in m)
    
    
    #   Example:
#   -------
#   #-- Generate a focused pressure field with a phased-array transducer
#   # Phased-array @ 2.7 MHz:
#   param = getparam('P4-2v');
#   # Focus position:
#   x0 = 2e-2; z0 = 5e-2;
#   # TX time delays:
#   dels = txdelay(x0,z0,param);
#   # Grid:
#   x = linspace(-4e-2,4e-2,200);
#   z = linspace(param.pitch,10e-2,200);
#   [x,z] = meshgrid(x,z);
#   y = zeros(size(x));
#   # RMS pressure field:
#   P = pfield(x,y,z,dels,param);
#   imagesc(x(1,:)*1e2,z(:,1)*1e2,20*log10(P/max(P(:))))
#   hold on, plot(x0*1e2,z0*1e2,'k*'), hold off
#   colormap hot, axis equal tight
#   caxis([-20 0])
#   c = colorbar;
#   c.YTickLabel{end} = '0 dB';
#   xlabel('[cm]')
    
    
    #   This function is part of <a
#   href="matlab:web('https://www.biomecardio.com/MUST')">MUST</a> (Matlab UltraSound Toolbox).
#   MUST (c) 2020 Damien Garcia, LGPL-3.0-or-later
    
    #   See also TXDELAY, PFIELD, SIMUS, GETPULSE.
    
    #   -- Damien Garcia -- 2015/03, last update: 2020/07
#   website: <a
#   href="matlab:web('https://www.biomecardio.com')">www.BiomeCardio.com</a>
    param = utils.Param()
    probe = probe.upper()
  
    
    # from computeTrans.m (Verasonics, version post Aug 2019)
    if 'L11-5V' == probe:
        # --- L11-5v (Verasonics) ---
        param.fc = 7600000.0
        param.kerf = 3e-05
        param.width = 0.00027
        param.pitch = 0.0003
        param.Nelements = 128
        param.bandwidth = 77
        param.radius = np.inf
        param.height = 0.005
        param.focus = 0.018
    elif 'L12-3V' == probe:
        # --- L12-3v (Verasonics) ---
        param.fc = 7540000.0
        param.kerf = 3e-05
        param.width = 0.00017
        param.pitch = 0.0002
        param.Nelements = 192
        param.bandwidth = 93
        param.radius = np.inf
        param.height = 0.005
        param.focus = 0.02
    elif 'C5-2V' == probe:
        # --- C5-2v (Verasonics) ---
        param.fc = 3570000.0
        param.kerf = 4.8e-05
        param.width = 0.00046
        param.pitch = 0.000508
        param.Nelements = 128
        param.bandwidth = 79
        param.radius = 0.04957
        param.height = 0.0135
        param.focus = 0.06
    elif 'P4-2V' == probe:
        # --- P4-2v (Verasonics) ---
        param.fc = 2720000.0
        param.kerf = 5e-05
        param.width = 0.00025
        param.pitch = 0.0003
        param.Nelements = 64
        param.bandwidth = 74
        param.radius = np.inf
        param.height = 0.014
        param.focus = 0.06
        #--- From the OLD version of GETPARAM: ---#
    elif 'PA4-2/20' == probe:
        # --- PA4-2/20 ---
        param.fc = 2500000.0
        param.kerf = 5e-05
        param.pitch = 0.0003
        param.height = 0.014
        param.Nelements = 64
        param.bandwidth = 60
    elif 'L9-4/38' == (probe):
        # --- L9-4/38 ---
        param.fc = 5000000.0
        param.kerf = 3.5e-05
        param.pitch = 0.0003048
        param.height = 0.006
        param.Nelements = 128
        param.bandwidth = 65
    elif 'LA530' == (probe):
        # --- LA530 ---
        param.fc = 3000000.0
        width = 0.215 / 1000
        param.kerf = 0.03 / 1000
        param.pitch = width + param.kerf
        # element_height = 6/1000; # Height of element [m]
        param.Nelements = 192
    elif 'L14-5/38' == (probe):
        # --- L14-5/38 ---
        param.fc = 7200000.0
        param.kerf = 2.5e-05
        param.pitch = 0.0003048
        # height = 4e-3; # Height of element [m]
        param.Nelements = 128
        param.bandwidth = 70
    elif 'L14-5W/60' == (probe):
        # --- L14-5W/60 ---
        param.fc = 7500000.0
        param.kerf = 2.5e-05
        param.pitch = 0.000472
        # height = 4e-3; # Height of element [m]
        param.Nelements = 128
        param.bandwidth = 65
    elif 'P6-3' == (probe):
        # --- P6-3 ---
        param.fc = 4500000.0
        param.kerf = 2.5e-05
        param.pitch = 0.000218
        param.Nelements = 64
        param.bandwidth = 2 / 3 * 100
    else:
        raise Exception(np.array(['The probe ',probe,' is unknown. Should be one of [L11-5V, L12-3V, C5-2V, P4-2V, PA4-2/20, L9-4/38, LA530, L14-5/38, L14-5W/60, P6-3]']))

    return param