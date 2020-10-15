import numpy as np
import subprocess


def state_to_graph(s, P):

	# Create a list of tuples (src, dst, depth) from the matrix representing
	# the communication graph.
	graph = []
	for src in range(P):
		for dst in range(P):
			if s[src,dst] != 0:
				graph.append((src, dst, s[src,dst]))

	# Order list by "depth" value:
	graph.sort(key=lambda v: v[2])

	return graph



def get_reward (s, params):

	# Get benchmark parameters:
	bench_p = params["Benchmark"]

	# Generate file describing the environment for executing the collective:
	cfile = open(bench_p["opts"],"w+")

	cfile.write("# P\n")
	cfile.write(str(params["P"]) + "\n")

	cfile.write("# Root\n")
	cfile.write(str(params["root"]) + "\n")

	cfile.write("# m\n")
	cfile.write(str(params["m"]) + "\n")

	cfile.write("# S\n")
	cfile.write(str(params["S"]) + "\n")

	cfile.write("# M\n")
	cfile.write(str(params["M"]) + "\n")

	cfile.write("# Nodes\n")
	cfile.write("[" + ','.join(params["nodes"]) + "]\n")

	cfile.write("# Mapping\n")
	mapping_str = [str(x) for x in params["mapping"]]
	cfile.write("[" + ','.join(mapping_str) + "]\n")

	cfile.write("# Network\n")
	cfile.write(params["net"] + "\n")

	cfile.write("# Platform\n")
	cfile.write(bench_p["platform"] + "\n")

	cfile.write("# Collective\n")
	cfile.write(bench_p["collective"] + "\n")

	cfile.write("# Algorithm\n")
	cfile.write(bench_p["algorithm"] + "\n")

	cfile.write("# n_iter\n")
	cfile.write(str(bench_p["n_iter"]) + "\n")

	cfile.write("# Graph\n")
	state_str = state_to_graph(s, params["P"])
	graph_str = [str(x) for x in state_str]
	cfile.write("[" + ','.join(graph_str) + "]\n")

	cfile.close()

	# Invoke process: only one argument, previous file.
	proc = subprocess.run([ bench_p["exec"],
							"-f", bench_p["opts"] ],
							stdout=subprocess.PIPE,
							stderr=subprocess.PIPE,
							shell=False,
							universal_newlines=True
							)

	# Output is time of execution:
	# print("ERR: \n", proc.stderr)
	# print("OUT: \n", proc.stdout)
	time = float(proc.stdout)

	return time
