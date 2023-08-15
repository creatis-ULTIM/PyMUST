param.fc = 2.7e6;
param.bandwidth = 76;

param.Nelements = 1;


% Description of the geometry of the probe - We will explore this in the next seminar.
param.radius = inf;
param.kerf = 3e-05;
param.width = 0.00027;
param.pitch = 0.0003;
param.height = 0.005;

xScatterers = [0];
zScatterers = [1e-2];
RCScatterers = [1];
txdel = [0]';

[RF, RFspec] = simus(xScatterers, zScatterers, RCScatterers,  txdel, param);
%%

tic
mean(D, 3);
toc
%%
param = getparam('L11-5v');
tic
% Define the grid
x = linspace(-2.5e-2,2.5e-2,150);%# in m
z = linspace(0,5e-2,150); % in m
[x,z] = meshgrid(x,z);
y = [];
txdel = txdelay(param, 0);
P = pfield(x,y, z,txdel, param); % Copy the param, since otherwise TXapodization will be changed.
toc
%%

param = getparam('P4-2v')
xScatterers =[0, .5e-2]
zScatterers = [1e-2, 3e-2 ]
RCScatterers =  [1, 1]

activationDelaysTX = txdelay(param, 0)




RF= simus(xScatterers, zScatterers, RCScatterers,  activationDelaysTX, param);
