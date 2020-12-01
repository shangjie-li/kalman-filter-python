"""KF augment."""

import numpy as np

def kf_augment(xx, px, zu, v, r):
	if zu.shape[1] == 0:
		return xx, px
	else:
		len_zu = zu.shape[1]
		for j in range(0, len_zu):
			xx_new = np.matrix([[zu[0, j]],
			                    [0],
			                    [zu[1, j]],
			                    [- v],])
			s = np.matrix([[1, 0],
			               [0, 0],
			               [0, 1],
			               [0, 0],])
			px_new = s * r * s.T
			# Augment xx.
			if xx.shape[1] == 0:
				xx = xx_new
			else:
				xx = np.row_stack((xx, xx_new))
			# Augment px.
			if px.shape[1] == 0:
				px = px_new
			else:
				len_px = px.shape[1]
				ii = range(-4, 0)
				px = np.row_stack((px, np.zeros((4, len_px))))
				px = np.column_stack((px, np.zeros((len_px + 4, 4))))
				#~ px[range(-4, 0), range(-4, 0)] = px_new
				for p_i in range(0, 4):
					for p_j in range(0, 4):
						px[ii[p_i], ii[p_j]] = px_new[p_i, p_j]
		return xx, px
