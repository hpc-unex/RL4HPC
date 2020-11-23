import numpy  as np
import pandas as pd

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


def plot_file (file_names, graph_file=None, show=False):

	plt.figure(figsize=(12,8))
	plt.axis('on')

	for f_name in file_names:

		try:
			df = pd.read_csv(f_name, index_col=0, delimiter="\t", skiprows=16, names=["episode", "J", "t", "T", "reward", "baseline"])
		except:
			continue

		j = df["J"].to_numpy()

		# Reduce dimensionality
		X_AXIS = 100

		j = j.reshape((X_AXIS, -1))

		j_max  = j.max(axis=1)
		j_min  = j.min(axis=1)
		j_mean = j.mean(axis=1)
		j_std  = j.std(axis=1)

		plt.plot(np.arange(0, X_AXIS, 1), j_mean)
		plt.fill_between(np.arange(0, X_AXIS, 1), j_mean - j_std, j_mean + j_std, alpha=0.1)

		# Para que me imprima en los limites de J y se vea bien
		# diff10 = (np.max(j_mean) - np.min(j_mean)) * 0.1
		# plt.ylim((int(np.min(j_mean) - diff10), int(np.max(j_mean) + diff10)))

	plt.hlines(0, 0, X_AXIS, colors='r', linestyles='dashed')

	plt.legend(labels=file_names)
	plt.title("Cost per Episode")
	plt.xlabel('# Episode')
	plt.ylabel('J')

	if show == False:
		plt.savefig(graph_file, dpi=300)
	else:
		plt.show()




def plot_loss (J_history, title=None):

	j = np.array(J_history)
	#t = np.array(T_history)
	# d = D_history


	# Reduce dimensionality
	X_AXIS = 100
	# X_AXIS = len(J_history)

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


	# Para que me imprima en los limites de J y se vea bien
	diff10 = (np.max(j_mean) - np.min(j_mean)) * 0.1
	plt.ylim((int(np.min(j_mean) - diff10), int(np.max(j_mean) + diff10)))

	plt.hlines(0, 0, X_AXIS, colors='r', linestyles='solid')

	plt.title(title)
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
