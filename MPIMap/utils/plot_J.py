#!/usr/bin/env python

## MonteCarlo Policy Gradients (REINFORCE)
#
# Implementing MonteCarlo policy gradient algorithm
# (REINFORCE, [Williams, 1992]) to learn a mapping for a
# given communication graph representing an application.

import numpy  as np
import decimal

import sys
sys.path.append('../Env')
sys.path.append('../utils')

import time
import pdb
import json

from graph   import plot_graph
from plots   import plot_file




##########   MAIN   ##########

if len(sys.argv) < 3:
	print("ERROR: plot_J <output_graph_file> <input_data_file_1> . . . <input_data_file_N>")
	sys.exit(1)

output_graph = sys.argv[1]
input_files  = sys.argv[2:]

print(output_graph)
print(input_files)

plot_file (input_files, "cost", graph_file=output_graph)
