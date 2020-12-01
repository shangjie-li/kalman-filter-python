"""KF predict."""

import numpy as np

def kf_predict(xx, px, ti, q):
	if xx.shape[1] == 0:
		return xx, px
	else:
		f = np.matrix([[1, ti, 0, 0],
					   [0, 1, 0, 0],
					   [0, 0, 1, ti],
					   [0, 0, 0, 1],])
		g = np.matrix([[0.5 * ti ** 2, 0],
					   [ti, 0],
					   [0, 0.5 * ti ** 2],
					   [0, ti],])
		ff = f
		gg = g
		qq = q
		while ff.shape[0] < xx.shape[0]:
			ler = ff.shape[0]
			lec = ff.shape[1]
			ff = np.row_stack((ff, np.zeros((4, lec))))
			ff = np.column_stack((ff, np.zeros((ler + 4, 4))))
			#~ ff[range(-4, 0), range(-4, 0)] = f
			ii = range(-4, 0)
			jj = range(-4, 0)
			for f_i in range(0, 4):
				for f_j in range(0, 4):
					ff[ii[f_i], jj[f_j]] = f[f_i, f_j]
			gg = np.row_stack((gg, np.zeros((4, int(lec / 2)))))
			gg = np.column_stack((gg, np.zeros((ler + 4, 2))))
			#~ gg[range(-4, 0), range(-2, 0)] = g
			ii = range(-4, 0)
			jj = range(-2, 0)
			for g_i in range(0, 4):
				for g_j in range(0, 2):
					gg[ii[g_i], jj[g_j]] = g[g_i, g_j]
			qq = np.row_stack((qq, np.zeros((2, int(lec / 2)))))
			qq = np.column_stack((qq, np.zeros((int(ler / 2 + 2), 2))))
			#~ qq[range(-2, 0), range(-2, 0)] = q
			ii = range(-2, 0)
			jj = range(-2, 0)
			for q_i in range(0, 2):
				for q_j in range(0, 2):
					qq[ii[q_i], jj[q_j]] = q[q_i, q_j]
		xx = ff * xx
		px = ff * px * ff.T + gg * qq * gg.T
		return xx, px
