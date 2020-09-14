import numpy as np
import subprocess


def get_reward (s, M, P, iter=0):

	# Obtain reward as communication time
	M_str = str(M).strip('[]()')
	# print(M_str)

	proc = subprocess.run(["/Users/jarico/Documents/Investigacion/Software/RL/RL4HPC/MPIColls/code/Env/tLop/bcast",
						   "-P", str(P),
						   "-m", "1024",
						   "-M", M_str,
						   "-c", "MPI_Bcast",
						   "-a", "binomial",
						   "-n", "IB",
						   "-s", "CIEMAT"],
						  stdout=subprocess.PIPE,
						  stderr=subprocess.PIPE,
						  shell=False,
						  universal_newlines=True)

	"""
	output = proc.stdout
	err    = proc.stderr

	if (err != None):
	print("ERR: ", err)
	print("OUTPUT: ", output)
	"""

	time = float(proc.stdout)

	return time
