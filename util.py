import numpy as np

def normalize(W):
    W = W/np.linalg.norm(W, axis=0)
    return W

def randn(shape):
    return normalize(np.random.randn(shape[0], shape[1]))