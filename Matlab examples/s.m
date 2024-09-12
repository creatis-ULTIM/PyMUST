xs = [1.7, 1.3, 0.7, 0, -0.7, -1.3, -1.7, 0, -1, 1]*1e-2;
zs = [2.8, 3.2, 3.5, 3.6, 3.5, 3.2, 2.8, 2, 0.8, 0.8]*1e-2;
RC = ones(size(xs));

param = getparam('L11-5v');

param.attenuation = 0.;
txdel_0 = zeros(1,param.Nelements);
tic
[F,info, param] = mkmovie(xs, zs, RC,txdel_0,param, 'noAttenuation.gif');
toc