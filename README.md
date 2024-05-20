# PyMUST
This is a Python reimplementation of the MUST ultrasound toolbox for synthetic image generation and reconstruction (https://www.biomecardio.com/MUST/).

*Notice:* this is still under development, and might have bugs and errors. Also, even if results should be the same with Matlab version, small numerical differences are expected. If you find any bug/unconsistency with the matlab version, please open a github issue, or send an email to ({damien.garcia@creatis.insa-lyon.fr, gabriel.bernardino@upf.edu}). 

As a design decision, we have tried to keep syntax as close as possible with the matlab version, specially regarding the way functions are called. This has resulted in non-pythonic arguments (i.e., overuse of variable number of positional arguments). This allows to make use of Must documentation (https://www.biomecardio.com/MUST/documentation.html). Keep in mind that, since Python does not allow a changing number of returns, each function will output the maximum number of variables of the matlab version.

## Installation
The package works in OsX, Linux and Windows (but parallelism might not be available on Windows). We recommend installing it in a separate conda environment.

To install pymust with its dependencies (matplotlib, scipy, numpy), you can directly install from pip:
> pip install git+https://github.com/creatis-ULTIM/PyMUST.git

Alternatively, you can install from the test pypi using the following instruction:
> python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ PyMUST

## Main functions
Please refer to the Matlab documentation
- Transducer (getparam)
- Simulation (simus, pfield)
- Bmode and Dopplerformation from radiofrequencies (tgc, rf2iq, bmode, iq2doppler)

## Examples
In the folder "examples", you have python notebooks ilustrating the main functionalities of PyMUST. They are the same as the ones available in the Matlab version.

## Next steps
If there is a functionality that you would like to see, please open an issue .

- Port of 3D reconstruction and simulation.
- Update function documentation.
- Find computational bottlenecks, and optimise (possibly with C extensions).
