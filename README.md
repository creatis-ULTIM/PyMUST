# PyMUST
This is a Python reimplementation of the MUST ultrasound toolbox for synthetic image generation and reconstruction (https://www.biomecardio.com/MUST/).

*Notice:* this is still under development, and might have bugs and errors. Also, even if results should be the same with Matlab version, small numerical differences are expected. If you find any bug/unconsistency with the matlab version, please open a github issue, or send an email to ({damien.garcia@creatis.insa-lyon.fr, gabriel.bernardino@upf.edu}). 

As a design decision, we have tried to keep syntax as close as possible with the matlab version, specially regarding the way functions are called. This has resulted in non-pythonic arguments (i.e., overuse of variable number of positional arguments). This allows to make use of Must documentation (https://www.biomecardio.com/MUST/documentation.html). Keep in mind that, since Python does not allow a changing number of returns, each function will output the maximum number of variables of the matlab version.

## Installation
### Install from pip
> pip install pymust

### Download from github
To install a local version of pymust with its dependencies (matplotlib, scipy, numpy), download it, go to the main folder and then run:
The package works in OsX, Linux and Windows (but parallelism might not be available on Windows). We recommend installing it in a separate conda environment.

To install pymust with its dependencies (matplotlib, scipy, numpy), you can directly install from pip:
> pip install git+https://github.com/creatis-ULTIM/PyMUST.git

Alternatively, you can install from the test pypi using the following instruction:
> python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ PyMUST

## Main functions
Please refer to the Matlab documentation or examples for a full
- Transducer definition (getparam)
- Simulation (simus, pfield)
- Bmode and Dopplerformation from radiofrequencies (tgc, rf2iq, bmode, iq2doppler)

## Examples
In the folder "examples", you have python notebooks ilustrating the main functionalities of PyMUST. They are the same as the ones available in the Matlab version.

## Next steps
If there is a functionality that you would like to see, please open an issue.
- Port of 3D reconstruction and simulation.
- Update function documentation.
- Find computational bottlenecks, and optimise (possibly with C extensions).
- GPU acceleration
- Differentiable rendering.

## Citation
The proceeding for . If you use this library for your research, please cite:

- [D. Garcia, "Make the most of MUST, an open-source MATLAB UltraSound Toolbox", 2021 IEEE International Ultrasonics Symposium (IUS), 2021, pp. 1-4, doi: 10.1109/IUS52206.2021.9593605](https://www.biomecardio.com/publis/ius21.pdf)
- [D. Garcia, "SIMUS: an open-source simulator for medical ultrasound imaging. Part I: theory & examples", Computer Methods and Programs in Biomedicine, 218, 2022, p. 106726, doi: 10.1016/j.cmpb.2022.106726](https://www.biomecardio.com/publis/cmpb22.pdf)
- [A. Cigier, F. Varray and D. Garcia "SIMUS: an open-source simulator for medical ultrasound imaging. Part II: comparison with four simulators," Computer Methods and Programs in Biomedicine, 218, 2022, p. 106726, doi: 10.1016/j.cmpb.2022.106726.](www.biomecardio.com/publis/cmpb22a.pdf)
  
If you use the speckle tracking functionality:
- [V. Perrot and D. Garcia, "Back to basics in ultrasound velocimetry: tracking speckles by using a standard PIV algorithm", 2018 IEEE International Ultrasonics Symposium (IUS), 2018, pp. 206-212, doi: 10.1109/ULTSYM.2018.8579665](https://www.biomecardio.com/publis/ius18.pdf)

If you use `fsd beamforming:
- [V. Perrot, M. Polichetti, F. Varray and D. Garcia, "So you think you can DAS? A viewpoint on delay-and-sum beamforming," 'Ultrasonics, 111, 2021, p. 106309, doi: 10.1016/j.ultras.2020.106309] (https://www.biomecardio.com/publis/ultrasonics21.pdf)
If you use vector flow:
- [C. Madiena, J. Faurie, J. Porée and D. Garcia "Color and vector flow imaging in parallel ultrasound with sub-Nyquist sampling," IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control 65, 2018, pp. 795-802 10.1109/TUFFC.2018.2817885](https://hal.science/hal-01988025/)

## Acknowledgements
This work has been patially funded by Grant #RYC2022-035960-I funded by MICIU/AEI/ 10.13039/501100011033 and by the FSE+
![image](https://github.com/user-attachments/assets/31c21398-2c34-421c-a3a0-1e628ab5d0cd)
            

