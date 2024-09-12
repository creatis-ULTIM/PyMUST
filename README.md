# pymust

This is a Python reimplementation of the MUST ultrasound toolbox for synthetic image generation and reconstruction (https://www.biomecardio.com/MUST/).

*Notice:* this is still under development, and might have bugs and errors. Also, even if results should be the same with Matlab version, small numerical differences are expected. If you find any bug/unconsistency with the matlab version, please open a github issue, or send an email to ({damien.garcia@creatis.insa-lyon.fr, gabriel.bernardino@upf.edu}). 

As a design decision, we have tried to keep syntax as close as possible with the matlab version, specially regarding the way functions are called. This has resulted in non-pythonic arguments (i.e., overuse of variable number of positional arguments). This allows to make use of Must documentation (https://www.biomecardio.com/MUST/documentation.html). Keep in mind that, since Python does not allow a changing number of returns, each function will output the maximum number of variables of the matlab version.

## Installation
### Install from pip
> pip install pymust

### Download from github
To install a local version of pymust with its dependencies (matplotlib, scipy, numpy), download it, go to the main folder and then run:

> pip install -e .

We recommend installing it in a separate conda environment.


## Next steps

- Port of 3D reconstruction and simulation.
- Port of sptrack and smoothn.
- Parallelize the simus and pfiel computations.
- Update function documentation.
- Add examples.
- Find computational bottlenecks, and optimise (possibly with C extensions).
