"""Associate."""

import numpy as np
import math

def associate(za, id_za, zu, z, xx, gate_associate):
	if z.shape[1] == 0:
		return za, id_za, zu
	else:
		for j in range(0, z.shape[1]):
			distance_m = float("inf")
			id_associate_best = float("inf")
			for k in range(0, int(xx.shape[0] / 4)):
				xx_x = xx[4 * k, 0]
				xx_y = xx[4 * k + 2, 0]
				z_x = z[0, j]
				z_y = z[1, j]
				dd = (z_x - xx_x) ** 2 + (z_y - xx_y) ** 2
				distance = math.sqrt(dd)
				if distance < gate_associate and distance < distance_m:
					distance_m = distance
					id_associate_best = k
			# Association accomplished.
			if id_associate_best != float("inf"):
				za_new = z[:, [j]]
				id_za_new = id_associate_best
				if za.shape[1] == 0:
					za = za_new
					id_za = [id_za_new]
				else:
					za = np.column_stack((za, za_new))
					id_za.append(id_za_new)
			# Association failed.
			else:
				zu_new = z[:, [j]]
				if zu.shape[1] == 0:
					zu = zu_new
				else:
					zu = np.column_stack((zu, zu_new))
		return za, id_za, zu
