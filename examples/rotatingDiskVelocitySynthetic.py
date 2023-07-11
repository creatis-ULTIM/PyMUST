# %%
import numpy as np
import pymust # To install it, use pip -e . install, orcopy the src/pymust folder to your project
import pymust.utils
import matplotlib.pyplot as plt
import scipy, scipy.io
import matplotlib, tqdm, tqdm.notebook, time
import cProfile, pstats


# %% [markdown]
# # Test disk 
# Here I generate a rotating disk, I test simus, txdelay, dasmtx, rf2iq and bmode

# %%
def rotatePoints(x, y, x0, y0, theta):
    """
    Rotate points (x,y) around (x0,y0) by theta radians
    """
    x = x - x0
    y = y - y0
    x1 = x * np.cos(theta) - y * np.sin(theta) + x0
    y1 = x * np.sin(theta) + y * np.cos(theta) + y0
    return x1, y1

if __name__ == '__main__':    

    profiler = cProfile.Profile()
    profiler.enable()

    # %%
    param = pymust.getparam('P4-2v')
    nPoints = 20000
    xs = np.random.rand(1,nPoints)*12e-2-6e-2
    zs = np.random.rand(1,nPoints)*12e-2

    centerDisk = 0.05
    idx = np.hypot(xs,zs-centerDisk)<2e-2 # First disk
    idx2 = np.hypot(xs,zs-.035)< 5e-3 # Second disk

    RC = np.random.rand(*xs.shape)  # reflection coefficients

    # Add reflectiion to both spheres
    RC[idx] += 1
    RC[idx2] += 2


    # Rotating disk velocity
    rotation_frequency = .5
    w = 2 * np.pi  * rotation_frequency#1 Hz = 2 pi rads
    nreps = 5
    param.PRP = 1e-3


    # %%
    t = time.time()
    for i in tqdm.tqdm(range(nreps)):
        options = pymust.utils.Options()
        options.ParPoolMode = 'process'
        options.ParPool_NumWorkers = 12
        options.dBThresh = -6
        options.ParPool = False

        xs[idx], zs[idx] = rotatePoints(xs[idx], zs[idx], 0, centerDisk,  w *  param.PRP)
        width = 60/180*np.pi; # width angle in rad
        txdel = pymust.txdelay(param,0,width) # in s
        RF, RF_spectrum = pymust.simus(xs,zs,RC,txdel,param, options)
        if i ==0:
            IQ = np.zeros([RF.shape[0], RF.shape[1], nreps], dtype = np.complex128)
        IQ[:, :, i] = pymust.rf2iq(RF,param)
    print('elapsed time = ', time.time() - t)

    # %%
    x,z = pymust.impolgrid(np.array([256, 256]),
                            10e-2, 
                            np.pi/3,
                            param)
    Mdas = pymust.dasmtx(IQ[:,:,0],x,z,param)
    IQb = np.zeros((x.shape[0], x.shape[1], nreps), dtype = np.complex128)
    for i in tqdm.notebook.tqdm(range(nreps)):
        IQb[:, :, i] = (Mdas@IQ[:,:,i].flatten(order = 'F')).reshape(x.shape, order = 'F')


    # %%
    b = pymust.bmode(IQb[:,:,0])
    fig = plt.figure(figsize=[5,5])
    ax = fig.add_axes([0.15,0.15,0.8,0.8])
    #fig.set_facecolor("black")
    ax.pcolormesh(z, x , b,edgecolors='face', cmap = 'gray')
    ax.set_facecolor('black')
    #ec='face' to avoid annoying gridding in pdf
    plt.savefig('carte_polar.png')


    # %%
    doppler_vel, doppler_var = pymust.iq2doppler(IQb,param)
    VN = pymust.getNyquistVelocity(param)

    # %%
    #fig = plt.figure(figsize=[5,5])
    #ax = fig.add_axes([0.15,0.15,0.8,0.8])
    #fig.set_facecolor("black")
    plt.pcolormesh(z, x , b,edgecolors='face', cmap = 'gray')
    plt.axis('equal')

    cm = plt.pcolormesh(z, x , doppler_vel,edgecolors='face', cmap = pymust.getDopplerColorMap().cmap, alpha = doppler_var/np.max(doppler_var), vmin = -VN, vmax = VN)
    ax.set_facecolor('black')
    #ec='face' to avoid annoying gridding in pdf
    cbar = plt.colorbar(cm)
    cbar.ax.set_ylabel('Velocity [m/s]')
    ax = plt.gca()
    ax.set_facecolor('black')

    plt.savefig('carte_polar.png')


    # %%
    fig = plt.figure(figsize=[5,5])
    ax = fig.add_axes([0.15,0.15,0.8,0.8])
    #fig.set_facecolor("black")
    ax.pcolormesh(z, x , b,edgecolors='face', cmap = 'gray')

    cm = ax.pcolormesh(z, x , doppler_var,edgecolors='face', cmap = pymust.getDopplerColorMap().cmap, )
    ax.set_facecolor('black')
    #ec='face' to avoid annoying gridding in pdf
    cbar = plt.colorbar(cm)
    cbar.ax.set_ylabel('Variance')
    plt.title('Doppler variance')
    #plt.savefig('carte_polar.png')


    # %%

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')

    # Print the stats report
    stats.print_stats()  
    profiler.dump_stats('stats.prof')




