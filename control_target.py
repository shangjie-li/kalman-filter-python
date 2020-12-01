"""Let targets move randomly."""

import numpy as np
import math

def control_target(tar, i, ti, q):
	w = np.zeros((2, 1))
	w[0, 0] = math.sqrt(q[0, 0]) * np.random.randn(1)
	w[1, 0] = math.sqrt(q[1, 1]) * np.random.randn(1)
	f = np.matrix([[1, ti, 0, 0],
	               [0, 1, 0, 0],
	               [0, 0, 1, ti],
	               [0, 0, 0, 1],])
	g = np.matrix([[0.5 * ti ** 2, 0],
	               [ti, 0],
	               [0, 0.5 * ti ** 2],
	               [0, ti],])
	tar[:, [i]] = f * tar[:, [i - 1]] + g * w
	return tar
