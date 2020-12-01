"""Observe."""

import numpy as np
import math

def observe(z, tar, i, arange, r):
	h = np.matrix([[1, 0, 0, 0],
	               [0, 0, 1, 0],])
	if tar[0, i] ** 2 + tar[2, i] ** 2 <= arange ** 2:
		v = np.zeros((2, 1))
		v[0, 0] = math.sqrt(r[0, 0]) * np.random.randn(1)
		v[1, 0] = math.sqrt(r[1, 1]) * np.random.randn(1)
		z_new = h * tar[:, [i]] + v
		if z.shape[1] == 0:
			z = z_new
		else:
			z = np.column_stack((z, z_new))
	return z
