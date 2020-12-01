"""Initialize location and velocity of targets."""

import numpy as np

def initialize_target(target, v, ni):
	target_new = np.zeros((4, ni))
	target_new[0, 0] = 40 * np.random.randn(1)
	target_new[1, 0] = 1 * np.random.randn(1)
	target_new[2, 0] = 1500 * np.random.rand(1) + 100
	target_new[3, 0] = 5 * np.random.randn(1) - v
	if target.shape[1] == 0:
		target = target_new
	else:
		target = np.row_stack((target, target_new))
	return target
