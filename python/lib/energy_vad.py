import numpy as np

import features
import gmm


def compute_vad(s, win_length=160, win_overlap=80, n_realignment=5, threshold=0.3):
    # power signal for energy computation
    s = s**2

    # frame signal with overlap
    F = features.framing(s, win_length, win_length - win_overlap) 

    # sum frames to get energy
    E = F.sum(axis=1)

    # E = np.sqrt(E)
    # E = np.log(E)

    # normalize the energy
    E -= E.mean()
    E /= E.std()


    # initialization
    mm = np.array((-1.00, 0.00, 1.00))[:, np.newaxis]
    ee = np.array(( 1.00, 1.00, 1.00))[:, np.newaxis]
    ww = np.array(( 0.33, 0.33, 0.33))

    GMM = gmm.gmm_eval_prep(ww, mm, ee)

    E = E[:,np.newaxis]

    for i in xrange(n_realignment):
        # collect GMM statistics
        llh, N, F, S = gmm.gmm_eval(E, GMM, return_accums=2)

        # update model
        ww, mm, ee   = gmm.gmm_update(N, F, S)

        # wrap model
        GMM = gmm.gmm_eval_prep(ww, mm, ee)

    # evaluate the gmm llhs
    llhs = gmm.gmm_llhs(E, GMM)

    llh  = gmm.logsumexp(llhs, axis=1)[:, np.newaxis]

    llhs = np.exp(llhs - llh)

    out  = np.zeros(llhs.shape[0], dtype=np.bool)
    out[llhs[:,0] < threshold] = True

    return out
    

