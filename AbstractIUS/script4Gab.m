
I = rgb2gray(imread('carotid.jpg'));
I = smoothn(double(I),100);
I = I-min(I(:));
%%
param = getparam('L11-5v');
param.attenuation = 0.5;
L = (param.Nelements-1)*param.pitch;

DistanceFactor = 1;
[xs,~,zs,RC] = genscat([5e-2 NaN],1540/param.fc*DistanceFactor,I,.4);

DR = 50;
dels = zeros(1,param.Nelements);
%[xs,zs,RC] = rmscat(xs,zs,RC,dels,param,DR);
[RF,param] = simus(xs,zs,RC,dels,param);

IQ = rf2iq(RF,param.fs,param.fc);
[xi,zi] = meshgrid(linspace(-L/2,L/2,256),linspace(eps,3e-2,round(256/L*3e-2)));

param.fnumber = [];
IQb = das(IQ,xi,zi,zeros(1,size(IQ,2)),param);
B = bmode(IQb);
imshow(B)