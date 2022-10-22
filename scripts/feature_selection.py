from implementations import *
from scripts.helpers import *
from scripts.dataset import *


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
    
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.
        
    Returns:
        poly: numpy array of shape (N,d+1)
    """
    poly_arr = np.hstack([np.power(x, d).reshape(-1, 1) for d in range(degree+1)])
    return poly_arr

