import scipy.io
import numpy as np
def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``."""
    m = scipy.io.loadmat('gen.mat')
    a = m['X1']
    b = m['Y1']
    b =np.ravel(b)
    return (a,b)

def vectorized_result(j):
    """Return a 2-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert the gender label
     into a corresponding desired output from the neural
    network."""
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e
load_data_wrapper()
