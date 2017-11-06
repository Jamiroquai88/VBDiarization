import numpy as np

from vbdiar.features.features import Features
from vbdiar.ivectors.gmm import GMM


def compute_vad(s, win_length=160, win_overlap=80, n_realignment=5, threshold=0.3):
    # power signal for energy computation
    s = s**2

    # frame signal with overlap
    F = Features.framing(s, win_length, win_length - win_overlap)

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

    GMM = GMM.gmm_eval_prep(ww, mm, ee)

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


def load_vad_lab_as_bool_vec(lab_file):
    lab_cont = np.atleast_2d(np.loadtxt(lab_file, dtype=object))

    if lab_cont.shape[1] == 0:
        return np.empty(0), 0, 0

    # else:
    #     lab_cont = lab_cont.reshape((-1,lab_cont.shape[0]))

    if lab_cont.shape[1] == 3:
        lab_cont = lab_cont[lab_cont[:, 2] == 'sp', :][:, [0, 1]]

    n_regions = lab_cont.shape[0]
    ii = 0
    while True:
        try:
            start1, end1 = float(lab_cont[ii][0]), float(lab_cont[ii][1])
            jj = ii + 1
            start2, end2 = float(lab_cont[jj][0]), float(lab_cont[jj][1])
            if end1 >= start2:
                lab_cont = np.delete(lab_cont, ii, axis=0)
                ii -= 1
                lab_cont[jj - 1][0] = str(start1)
                lab_cont[jj - 1][1] = str(max(end1, end2))
            ii += 1
        except IndexError:
            break

    vad = np.round(np.atleast_2d(lab_cont).astype(np.float).T * 100).astype(np.int)
    vad[1] += 1  # Paja's bug!!!

    if not vad.size:
        return np.empty(0, dtype=bool)

    npc1 = np.c_[np.zeros_like(vad[0], dtype=bool), np.ones_like(vad[0], dtype=bool)]
    npc2 = np.c_[vad[0] - np.r_[0, vad[1, :-1]], vad[1] - vad[0]]

    out = np.repeat(npc1, npc2.flat)

    n_frames = sum(out)

    return out, n_regions, n_frames
    

