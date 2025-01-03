#!/usr/bin/env python

from distutils.core import setup
import subprocess

def _get_version_hash():
    """Talk to git and find out the tag/hash of our latest commit"""
    try:
        p = subprocess.Popen(["git", "describe",
                            "--tags", "--dirty", "--always"],
                            stdout=subprocess.PIPE)
    except EnvironmentError:
        print("Couldn't run git to get a version number for setup.py")
        return
    ver = p.communicate()[0]
    if isinstance(ver, bytes):
        ver = ver.decode('ascii')
    return ver.strip()

setup(name='pymust',
      description='Python port of the MUST toolbox for ultrasound signal processing and generation of simulated images.',
      author='Gabriel Bernardino (Python port), Damien Garcia (original matlab code)',
      author_email='gabriel.bernardino1@gmail.com',
      url='https://www.biomecardio.com/en//',
      #version= '0.1.3',
      packages=['pymust'],
      license = 'GNU Lesser General Public License v3.0 (LGPL v3)',
      package_dir={'':'src'}
     )
