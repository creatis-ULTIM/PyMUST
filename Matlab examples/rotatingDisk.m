addpath('~/Downloads/MUST')
param = getparam('P4-2v');
nPoints = 200000;
xs = rand(1,nPoints)*12e-2-6e-2;
zs =rand(1,nPoints)*12e-2;

centerDisk = 0.05;
idx = hypot(xs,zs-centerDisk)<2e-2; % First disk
idx2 = hypot(xs,zs-.035)< 5e-3; % Second disk

RC = rand(size(xs));  % reflection coefficients

% Add reflectiion to both spheres
RC(idx) = RC(idx)  + 1;
RC(idx2) = RC(idx2)+ 2;
clear IQ;
%%--
%%

% Rotating disk velocity
rotation_frequency = .5;
w = 2 * pi  * rotation_frequency; %1 Hz = 2 pi rads
nreps = 5;
param.PRP = 1e-3;
tic
for i  = 1:nreps
    options.dBThresh = -6;
    options.ParPool = true;

    [xs_rot, zs_rot] = rotatePoints(xs(idx), zs(idx), 0, centerDisk,  w *  param.PRP);
    xs(idx) = xs_rot;
    zs(idx) = zs_rot;
    width = 60/180*pi; %width angle in rad
    txdel = txdelay(param,0,width); % in s
    [RF, param, RF_spectrum] = simus(xs,zs,RC,txdel,param, options);
    param.fs = 4 * param.fc;
    IQ(:, :, i) = rf2iq(RF,param);
end 
toc

% Beamforming
[x, z] = impolgrid(128,10e-2,pi/2,param); 
M = dasmtx(IQ(:,:,1),x, z,txdel, param);
IQ_r = reshape(IQ, [size(M, 2), size(IQ, 3)]);
IQb = M*IQ_r;
IQb = reshape(IQb, [size(x,1), size(x,2), size(IQ, 3)]);
[v, var] = iq2doppler(IQb, param);