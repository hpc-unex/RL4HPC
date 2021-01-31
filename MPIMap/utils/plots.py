import numpy  as np
import pandas as pd

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


def plot_file (file_names, sdata, graph_file=None, show=False):

	plt.figure(figsize=(12,8))
	plt.axis('on')

	# Reduce dimensionality
	X_AXIS = 100

	for f_name in file_names:

		# print("Fichero: ", f_name)

		try:
			df = pd.read_csv(f_name, index_col=0, delimiter="#", skiprows=15, names=["episode", "J", "t", "T", "reward", "baseline", "actions"])
		except:
			continue

		# print(df)

		if sdata == "cost":
			j = df["J"].to_numpy()
		elif sdata == "reward":
			j = df["reward"].to_numpy()
		else:
			print("ERROR: in data to plot")
			return

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
	if sdata == "cost":
		plt.title("Cost per Episode")
		plt.xlabel('# Episode')
		plt.ylabel('J')
	elif sdata == "reward":
		plt.title("Reward per Episode")
		plt.xlabel('# Episode')
		plt.ylabel('Reward')


	if show == False:
		plt.savefig(graph_file, dpi=300)
	else:
		plt.show()
