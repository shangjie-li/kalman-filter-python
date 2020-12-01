"""KF update."""

import numpy as np

def kf_update(xx, px, za, id_za, r):
	if za.shape[1] == 0:
		return xx, px
	else:
		h = np.matrix([[1, 0, 0, 0],
					   [0, 0, 1, 0],])
		len_xx = xx.shape[0]
		len_za = za.shape[1]
		hh = np.zeros((2 * len_za, len_xx))
		zz = np.zeros((2 * len_za, 1))
		rr = np.zeros((2 * len_za, 2 * len_za))
		for j in range(0, len_za):
			ii = [2 * j, 2 * j + 1]
			jj = range(4 * id_za[j], 4 * id_za[j] + 4)
			#~ hh[ii, jj] = h
			for h_i in range(0, 2):
				for h_j in range(0, 4):
					hh[ii[h_i], jj[h_j]] = h[h_i, h_j]
			zz[ii, :] = za[:, [j]] - hh[ii, :] * xx
			#~ rr[ii, ii] = r
			for r_i in range(0, 2):
				for r_j in range(0, 2):
					rr[ii[r_i], ii[r_j]] = r[r_i, r_j]
		kk = px * hh.T * np.linalg.inv(hh * px * hh.T + rr)
		xx = xx + kk * zz
		px = px - kk * hh * px
		return xx, px
