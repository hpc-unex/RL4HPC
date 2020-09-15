import numpy as np
import subprocess


def get_reward (s, params):

	# Get benchmark parameters:
	bench_p = params["Benchmark"]

	# Generate file describing the environment for executing the collective:
	# TODO
	cfile = open(bench_p["opts"],"w+")
	cfile.write("# P\n")
	cfile.write("# root\n")
	cfile.write("# m\n")
	cfile.write("# S\n")
	cfile.write("# M\n")
	cfile.write("# nodes\n")
	cfile.write("# mapping\n")
	cfile.write("# network\n")
	cfile.write("# platform\n")
	cfile.write("# Collective\n")
	cfile.write("# Algorithm\n")
	cfile.write("# Graph\n")
	cfile.close()

	# Invoke process: only one argument, previous file.
	proc = subprocess.run([ bench_p["exec"],
							"-f", bench_p["opts"] ],
							stdout=subprocess.PIPE,
							stderr=subprocess.PIPE,
							shell=False,
							universal_newlines=True
							)

	"""
	output = proc.stdout
	err    = proc.stderr

	if (err != None):
	print("ERR: ", err)
	print("OUTPUT: ", output)
	"""

	# Output is time of execution:
	time = float(proc.stdout)

	# print("TIME: ", time)

	return time
