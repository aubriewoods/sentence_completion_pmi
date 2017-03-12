import numpy as np
from numpy import array
from copy import copy

def pmi(m, positive=False, discounting=False):    
    # Joint probability table:
    p = m.mat / np.sum(m.mat, axis=None)
	
    # PMI with positive option:
    colprobs = np.sum(p, axis=0)
    np_pmi_log = np.vectorize((lambda x : pmi_log(x, positive=positive)))
    p = array([np_pmi_log(row / (np.sum(row)*colprobs)) for row in p])
	
    # Optional discounting:
    if discounting:
        colsums = np.sum(m.mat, axis=0)
        fmatrix = m.mat / (m.mat + 1)
        dmatrix = array([pmi_discounting_row_func(row, colsums) for row in m.mat])        
        p *= fmatrix * dmatrix
    reweighted = copy(m)
    reweighted.mat = p
    return reweighted    

    
def pmi_log(x, positive=False):
    if (x <= 0.0): #log(0) is undefined
        return 0.0
    else:
        x = np.log2(x)
        if (positive and x < 0.0):
            x = 0.0
        return x

        
def pmi_discounting_row_func(row, colsums):
    mincontext = np.minimum(np.sum(row), colsums)
    return mincontext / (mincontext + 1)