import numpy as np
import pandas as pd

def get_R_nrm(R, d=None):
    V = R.shape[0]
    R = R.astype(float)
    R += np.eye(V) * R.sum(axis=0) # Add self-edge equal to sum of victories
    Rt = R.T
    idx = Rt.sum(axis=0) > 0
    Rt[:,idx] /= Rt[:,idx].sum(axis=0)
    R_nrm = Rt.T
    
    if d:
        assert d > 0 and d < 1
        R_nrm *= d
        R_nrm += (1 - d) / V
        
    return R_nrm

# Eigenvector methods
def power_method(R_nrm):
    V = R_nrm.shape[0]
    x = np.zeros(V) + 1. / V
    prev = x
    while True:
        prev = x
        x = x.dot(R_nrm)
        if np.linalg.norm((x - prev)) < 0.0001:
            break
    return pd.Series(x).sort_values(ascending=False)

def np_method(R_nrm):
    eigval, eigvec = np.linalg.eig(R_nrm.T)
    eigval = pd.Series(eigval)
    ev = pd.Series(eigvec[:,eigval.idxmax()])
    return (ev / ev.sum()).sort_values(ascending=False) # Normalize to simplex

# Error metrics
def get_error(ordering):
    V = len(ordering)
    return sum(np.abs(ordering.index - np.arange(V)[::-1]))

# Tournament entropy
def te(R):
    in_nrm = R.sum(axis=0) / R.sum().astype(float)
    in_nrm = in_nrm[in_nrm > 0]
    return -(in_nrm * np.log2(in_nrm)).sum()

def te_max(V):
    return - (V * ((1. / V) * np.log2(1. / V)))