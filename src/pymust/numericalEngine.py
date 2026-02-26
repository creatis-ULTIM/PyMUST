from hashlib import new
import numpy, importlib



class NumericalEngine:
    """
    Class implementing a numerical engine, that allows to potentially different backend libraries and devices, 
    and call them using the numpy syntax. It will allow to run with minimal dependencies installed.
    """
    def __init__(self, backend = 'numpy', device = 'cpu'):
        self.device = device
        if backend == 'numpy':
            self.isNumpy = True
            self.backend = numpy
            self.backend_name = 'numpy'
            if self.device != 'cpu':
                raise ValueError('Numpy backend only supports cpu device')  
        elif backend == 'torch':
            ar = importlib.import_module('autoray')
            self.isNumpy = False
            self.backend_name = 'torch'
            self.backend = ar.get_namespace('torch')
            self.backend.set_default_device(device)
        else:
            raise ValueError('Unsupported backend')
        
    def to_numpy(self, x):
        if self.isNumpy:
            return x
        elif self.device != 'cpu':
            return x.detach().to('cpu').numpy()
        else:
            return x.detach().numpy()
        
    def to_backend(self, x):
        if self.isNumpy:
            return x
        else:
            return self.backend.asarray(x, device=self.device)
        
    def __deepcopy__(self, memo):
        new = NumericalEngine(self.backend_name, self.device)
        memo[id(self)] = new
        return new

NumpyEngine = NumericalEngine() 
    