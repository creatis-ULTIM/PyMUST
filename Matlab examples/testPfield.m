p = gcp(); % If no pool, do not create new one.
if isempty(p)
    poolsize = 0;
else
    poolsize = p.NumWorkers
end
%%
I = rgb2gray(imread('heart.jpg'));
% Pseudorandom distribution of scatterers (depth is 15 cm)
fc = 2e6;
[xs,y,zs,RC] = genscat([nan, 15e-2],1540/fc,I);


param.fc = 2.7e6;
param.bandwidth = 76;

param.Nelements = 1;
%%
tic
param = getparam('L11-5v');
param.attenuation = 0.5;
OPTIONS.ParPool = true;
param.fs = 4 * param.fc;


L = (param.Nelements-1)*param.pitch;
param.TXdelay = txdelay(param,deg2rad(10));
RF= simus(xs,zs,RC,param.TXdelay,param, OPTIONS);
RF = tgc(RF);
IQ = rf2iq(RF, param.fs,param.fc);

param.fnumber = [];
[xi_linear,zi_linear] = impolgrid([100, 100],15e-2,deg2rad(120),param);
M = dasmtx(IQ,xi_linear,zi_linear,param);
IQb = M*reshape(IQ,[], 1);
IQb = reshape(IQb,size(xi_linear));

B_linear = bmode(IQb);
toc
tic
P_linear = pfield(xi_linear, [], zi_linear, param.TXdelay, param);
toc

%%
tic
param = getparam('P6-3');
param.attenuation = 0.5;
OPTIONS.ParPool = true;
param.fs = 4 * param.fc;


L = (param.Nelements-1)*param.pitch;
param.TXdelay = txdelay(param, -pi/12, pi/3);
RF= simus(xs,zs,RC,param.TXdelay,param, OPTIONS);
RF = tgc(RF);
IQ = rf2iq(RF, param.fs,param.fc);

param.fnumber = [];
[xi_linear,zi_linear] = impolgrid([200, 200],15e-2,deg2rad(120),param);
M = dasmtx(IQ,xi_linear,zi_linear,param);
IQb = M*reshape(IQ,[], 1);
IQb = reshape(IQb,size(xi_linear));

B_linear = bmode(IQb);
toc
tic
P_linear = pfield(xi_linear, [], zi_linear, param.TXdelay, param);
toc
%%
tic
param = getparam('C5-2V');
param.attenuation = 0.5;
OPTIONS.ParPool = true;
param.fs = 4 * param.fc;


L = (param.Nelements-1)*param.pitch;
param.TXdelay = txdelay(param, 0);
RF= simus(xs,zs,RC,param.TXdelay,param, OPTIONS);
RF = tgc(RF);
IQ = rf2iq(RF, param.fs,param.fc);

param.fnumber = [];
[xi_linear,zi_linear] = impolgrid([200, 100],15e-2, param);
M = dasmtx(IQ,xi_linear,zi_linear,param);
IQb = M*reshape(IQ,[], 1);
IQb = reshape(IQb,size(xi_linear));

B_linear = bmode(IQb);
toc
tic
P_linear = pfield(xi_linear, [], zi_linear, param.TXdelay, param);
toc
