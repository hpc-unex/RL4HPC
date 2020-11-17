#!/usr/bin/env python

## MonteCarlo Policy Gradients (REINFORCE)
#
# Implementing MonteCarlo policy gradient algorithm
# (REINFORCE, [Williams, 1992]) to learn a mapping for a
# given communication graph representing an application.

import numpy  as np
import pandas as pd
import decimal

import sys
sys.path.append('../Env')
sys.path.append('../utils')

import time
import pdb
import json

from graph         import plot_graph

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout




def plot_file (file_names, graph_file=None, show=False):

	plt.figure(figsize=(12,8))
	plt.axis('on')

	for f_name in file_names:

		try:
			df = pd.read_csv(f_name, index_col=0, delimiter="\t", skiprows=16, names=["episode", "J", "t", "T", "n_intra", "n_inter", "n_errors"])
		except:
			continue

		### TEMPORAL
		j = df["n_errors"].to_numpy() - df["n_inter"].to_numpy()

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
	plt.title("Reward per Episode")
	plt.xlabel('# Episode')
	plt.ylabel('R')

	if show == False:
		plt.savefig(graph_file, dpi=300)
	else:
		plt.show()






##########   MAIN   ##########

if len(sys.argv) < 3:
	print("ERROR: plot_R <output_graph_file> <input_data_file_1> . . . <input_data_file_N>")
	sys.exit(1)

output_graph = sys.argv[1]
input_files  = sys.argv[2:]

print(output_graph)
print(input_files)

plot_file (input_files, graph_file=output_graph)
