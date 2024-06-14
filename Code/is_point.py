import numpy as np
from numpy.linalg import solve

def _shift(boundary: np.array, target: np.array) -> (np.array, np.array):
    """
    Helper function for `is_point`.
    Shifts all vectors so that `boundary[0]` is the origin of a new space.
    """
    A = np.array(boundary[1:4] - boundary[0]).T
    b = np.array(target-boundary[0]).T
    return A, b

def is_point(boundary, target):
    """
    Classifies a point `target` as being inside a given boundary or not inside the given boundary.
    """
    
    A, b = _shift(boundary, target)
    mask = np.ones(b[0].size, dtype=bool)
    w = solve(A,b)
    mask[np.any(w<0,axis=0)] = False
    mask[w.sum(axis=0)>1] = False
    
    return mask
