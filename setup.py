#!/usr/bin/env python

from distutils.core import setup
setup(name='pymust',
      version='0.1.2',
      description='Python port of the MUST toolbox for ultrasound signal processing and generation of simulated images.',
      author='Gabriel Bernardino (Python port), Damien Garcia (original matlab code)',
      author_email='gabriel.bernardino1@gmail.com',
      url='https://www.biomecardio.com/en//',
      packages=['pymust'],
      license = 'GNU Lesser General Public License v3.0 (LGPL v3)',
      package_dir={'':'src'}
     )
