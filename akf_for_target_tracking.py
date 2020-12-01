"""Augment KF for target tracking."""

"""
The basic configuration parameters are as follow:
	ni - Number of iteration
	ti - Time interval
	sigmaax - Sigma of acceleration noise in the x direction
	sigmaay - Sigma of acceleration noise in the y direction
	q - Acceleration noise matrix
	sigmaox - Sigma of observation noise in the x direction
	sigmaoy - Sigma of observation noise in the y direction
	r - Observation noise matrix
	xtrue - Location of ego vehicle
	v - Velocity of ego vehicle
	arange - Range of observation
	nt - Number of targets
	gate_associate - Association distance
"""

import numpy as np
import math
import matplotlib.pyplot as plt

from initialize_target import initialize_target
from control_target import control_target
from kf_predict import kf_predict
from observe import observe
from associate import associate
from kf_update import kf_update
from kf_augment import kf_augment
from delete import delete

ni = 500
ti = 0.1
sigmaax = 1
sigmaay = 1
q = np.matrix([[sigmaax ** 2, 0],
               [0, sigmaay ** 2],])
sigmaox = 0.1
sigmaoy = 0.1
r = np.matrix([[sigmaox ** 2, 0],
               [0, sigmaoy ** 2],])
xtrue = [0, 0]
v = 30
arange = 75
nt = 30
gate_associate = 5

# Create a drawing window.
fig = plt.figure(1)

# Initialize state xx and covariance px.
xx, px = np.matrix([]), np.matrix([])

# Initialize location and velocity of targets.
# if target = np.matrix([]), then target.shape[1] = 0.
target = np.matrix([])
for n in range(0, nt):
	target = initialize_target(target, v, ni + 1)

for i in range(1, ni + 1):
	print("\nIteration:")
	print(i)
	
	# Let targets move randomly.
	for n in range(0, nt):
		target[range(4 * n, 4 * n + 4), :] = control_target(
				target[range(4 * n, 4 * n + 4), :], i, ti, q)
	
	# KF predict.
	xx, px = kf_predict(xx, px, ti, q)
	
	# Observe.
	# Initialize observation z.
	z = np.matrix([])
	for n in range(0, nt):
		z = observe(z, target[range(4 * n, 4 * n + 4), :], i, arange, r)
	
	# Associate.
	# Initialize za, id_za and zu.
	# za represents targets which have been observed and associated.
	# id_za stores indexes of targets in xx.
	# zu represents targets which have been observed but not associated.
	za, id_za, zu = np.matrix([]), [], np.matrix([])
	za, id_za, zu = associate(za, id_za, zu, z, xx, gate_associate)
	
	# KF update.
	xx, px = kf_update(xx, px, za, id_za, r)
	
	# Delete targets which are beyond the range of observation.
	xx, px = delete(xx, px, xtrue, arange)
	
	# KF augment.
	xx, px = kf_augment(xx, px, zu, v, r)
	
	print("xx:")
	print(xx)
	
	# Draw dynamically.
	plt.clf()
	plt.axis([-100, 100, -100, 100])
	plt.xlabel("Meter", fontsize=14)
	plt.ylabel("Meter", fontsize=14)
	
	# Draw ego vehicle.
	plt.scatter(xtrue[0], xtrue[1], c='blue', edgecolor='none', s=100)

	# Draw range of observation.
	theta = np.linspace(0, 2 * np.pi, 1000)
	arange_x = arange * np.cos(theta)
	arange_y = arange * np.sin(theta)
	plt.plot(arange_x, arange_y, '--', c='black', linewidth=1)
	
	# Draw real targets.
	for n in range(0, nt):
		tar_x = target[4 * n, i]
		tar_vx = target[4 * n + 1, i]
		tar_y = target[4 * n + 2, i]
		tar_vy = target[4 * n + 3, i] + v
		if math.fabs(tar_x) < 100 and math.fabs(tar_y) < 100:
			plt.scatter(tar_x, tar_y, c='black', edgecolor='none', s=25)
			text_real_ve = "Velocity: (" + str(round(tar_vx, 1)) + "," + str(round(tar_vy, 1)) + ")"
			plt.text(tar_x + 2, tar_y - 10, text_real_ve, color='lightgrey', fontsize=12)
	
	# Draw observation.
	if z.shape[1] == 0:
		pass
	else:
		for ob_i in range(0, z.shape[1]):
			ob_x = [xtrue[0], z[0, ob_i]]
			ob_y = [xtrue[1], z[1, ob_i]]
			plt.plot(ob_x, ob_y, c='black', linewidth=1)
	
	# Draw estimated targets.
	if xx.shape[1] == 0:
		pass
	else:
		for xx_n in range(0, int(xx.shape[0] / 4)):
			xx_x = xx[4 * xx_n, 0]
			xx_vx = xx[4 * xx_n + 1, 0]
			xx_y = xx[4 * xx_n + 2, 0]
			xx_vy = xx[4 * xx_n + 3, 0] + v
			plt.scatter(xx_x, xx_y, c='red', edgecolor='none', s=25)
			text_id = "ID: " + str(xx_n)
			text_lo = "Location: (" + str(round(xx_x, 1)) + "," + str(round(xx_y, 1)) + ")"
			text_ve = "Velocity: (" + str(round(xx_vx, 1)) + "," + str(round(xx_vy, 1)) + ")"
			plt.text(xx_x + 2, xx_y + 8, text_id, fontsize=12)
			plt.text(xx_x + 2, xx_y + 2, text_lo, fontsize=12)
			plt.text(xx_x + 2, xx_y - 4, text_ve, fontsize=12)
	
	# Draw legend.
	text_ego = "Ego Velocity: " + str(round(v, 1)) + "m/s"
	plt.text(-95, -95, text_ego, fontsize=12)
	plt.scatter(46, 90, c='black', edgecolor='none', s=25)
	plt.text(50, 88, "real targets", fontsize=12)
	plt.scatter(46, 80, c='red', edgecolor='none', s=25)
	plt.text(50, 78, "estimated targets", fontsize=12)
	
	# Close the drawing window after showing for a while.
	plt.pause(0.02)
	
	if i == ni:
		print("\nSimulation process finished!")
		break
