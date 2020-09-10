import numpy  as np

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


def plot_loss (J_history):

	j = np.array(J_history)
	#t = np.array(T_history)
	# d = D_history


	# Reduce dimensionality
	X_AXIS = 100

	j = j.reshape((X_AXIS, -1))

	j_max = j.max(axis=1)
	j_min = j.min(axis=1)
	j_mean = j.mean(axis=1)

	#t = t.reshape((X_AXIS, -1))
	#t = t.mean(axis=1)

	arr_mean = []
	for i in j_mean:
		arr_mean.append(i.item())


	j_ndarr = np.empty(j.shape, dtype=float)
	for i in range(0, j.shape[0]-1):
		for k in range(0, j.shape[1]-1):
			j_ndarr[i][k] = j[i][k].item()

	j_std = np.std(j_ndarr, axis=1)


	plt.figure(figsize=(12,8))
	plt.axis('on')

	# plt.plot(np.arange(0, X_AXIS, 1), j_mean, color='blue', marker='.')
	plt.plot(np.arange(0, X_AXIS, 1), j_mean)

	plt.fill_between(np.arange(0, X_AXIS, 1), arr_mean - j_std, arr_mean + j_std, alpha=0.1)

	plt.title("Cost function per Episode")
	plt.xlabel('# Episode')
	plt.ylabel('J')
	# plt.ylim(0)
	plt.show()

	"""
		plt.plot(np.arange(0, X_AXIS, 1), t, color='red', marker='.')
		plt.title("#steps in Trajectory per Episode")
		plt.xlabel('# Episode')
		plt.ylabel('T')
		plt.ylim(0)
		plt.show()
		
		plt.plot(np.arange(0, X_AXIS, 1), d, color='green', marker='.')
		plt.title("Depth of tree per Episode")
		plt.xlabel('# Episode')
		plt.ylabel('Depth')
		plt.ylim(0)
		plt.show()
		"""

