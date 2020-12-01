"""Delete targets which are beyond the range of observation."""

import numpy as np

def delete(xx, px, xtrue, arange):
	if xx.shape[1] == 0:
		return xx, px
	else:
		k = 0
		while k < int(xx.shape[0] / 4):
			xx_x = xx[4 * k, 0]
			xx_y = xx[4 * k + 2, 0]
			dd = (xx_x - xtrue[0]) ** 2 + (xx_y - xtrue[1]) ** 2
			if dd > arange ** 2:
				len_xx = xx.shape[0]
				if len_xx == 4:
					xx = np.matrix([])
					px = np.matrix([])
					break
				else:
					# Delete targets from xx.
					xx_copy = xx
					xx = np.zeros((len_xx - 4, 1))
					xx[range(0, 4 * k), :] = xx_copy[range(0, 4 * k), :]
					xx[range(4 * k, len_xx - 4), :] = xx_copy[range(4 * k + 4, len_xx), :]
					# Delete targets from px.
					px_copy = px
					px = np.zeros((len_xx - 4, len_xx))
					px[range(0, 4 * k), :] = px_copy[range(0, 4 * k), :]
					px[range(4 * k, len_xx - 4), :] = px_copy[range(4 * k + 4, len_xx), :]
					px_copy = px
					px = np.zeros((len_xx - 4, len_xx - 4))
					px[:, range(0, 4 * k)] = px_copy[:, range(0, 4 * k)]
					px[:, range(4 * k, len_xx - 4)] = px_copy[:, range(4 * k + 4, len_xx)]
					continue
			k += 1
		return xx, px
