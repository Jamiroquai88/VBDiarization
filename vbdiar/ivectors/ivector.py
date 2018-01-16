import numpy as np
#import scipy.linalg as spl


################################################################################
################################################################################
def estimate_i(Nt, Ft, v, VtV=None, I=None, out=None):
    v_dim   = v.shape[1]
    n_gauss = Nt.shape[1]
    n_data  = Nt.shape[0]

    # compute VtV if necessary
    if VtV is None:
        VtV = compute_VtV(v, n_gauss)

    # Allocate space for out i-vec if necessary
    if out is None: 
        out = np.empty((v_dim, n_data), dtype=v.dtype)

    # Construct eye if necessary
    if I is None:
        I = np.eye(v_dim, dtype=v.dtype)

    b   = np.dot(Ft, v).T
    L   = np.dot(Nt, VtV).reshape(-1, v_dim, v_dim) + I.reshape(-1, v_dim,v_dim)
    # out = spl.solve(L, b)
    # out = np.zeros(n_data,v_dim)
    for ii in xrange(n_data):
        out[:,ii] = np.linalg.solve(L[ii,:,:], b[:,ii])

    return out


################################################################################
################################################################################
def compute_VtV(v, n_gauss, out=None):

    v_dim   = v.shape[1]
    f_dim   = v.shape[0] / n_gauss

    # Allocate space if necessary
    if out is None: 
        out = np.empty((n_gauss, v_dim, v_dim), dtype=v.dtype)
    else:
        out = out.reshape((n_gauss, v_dim, v_dim))

    # reshape v to 
    v3d = v.reshape((n_gauss, f_dim, v_dim));

    for i in range(n_gauss):
        v_part = v3d[i,:,:]
        np.dot(v_part.T, v_part, out=out[i,:,:])

    out = out.reshape((n_gauss, v_dim*v_dim))
    return out


