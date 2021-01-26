import numpy as np
import subprocess


MAX_TIME = 1000.0

# A simple class to maintain a moving average baseline
class Baseline:

	baseline = 0.0
	episode  = 0

	def update(r):
		Baseline.episode += 1
		Baseline.baseline = Baseline.baseline + (1.0 / Baseline.episode) * (r - Baseline.baseline)

	def get():
		return Baseline.baseline






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



def get_reward (s, a, params):

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
	# mapping_str = [str(x) for x in params["nodes_procs"]]
	mapping_str = [str(x.item()) for x in a]
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


    # If capacity of nodes is overcome, return MAX_TIME
	# That is, we do not accept oversubscripting.
	"""
    int capacity[] = {4, 4, 4, 4};
    int req_capacity[pm.M];
    for (int i = 0; i < pm.M; i++) req_capacity[i] = 0;
    for (int i = 0; i < pm.P; i++) {
        req_capacity[pm.mapping[i]] += 1;
    }
    int f = 1;
    for (int i = 0; i < pm.M; i++) {
        if (req_capacity[i] != capacity[i]) {
            f += abs(req_capacity[i] - capacity[i]);
            // t += f * MAX_TIME;
        }
    }
    if (t >= MAX_TIME) {
        cout << t << endl;
        return 0;
    }
	"""

	P = params["P"]
	M = params["M"]
	unique_elements, counts_elements = np.unique(a.to("cpu"), return_counts=True)
	Q = P // M
	counts_elements = counts_elements - Q
	n_errors = (M - np.size(counts_elements)) * Q + np.sum(np.abs(counts_elements))
	if n_errors > 0:
		valid = False
		# time = n_errors * MAX_TIME
		time = MAX_TIME
		info = {"valid": valid, "reward": time, "baseline": Baseline.get()}

		return time, info



	# Invoke process: only one argument, previous file.
	proc = subprocess.run([ bench_p["exec"],
							"-f", bench_p["opts"] ],
							stdout=subprocess.PIPE,
							stderr=subprocess.PIPE,
							shell=False,
							universal_newlines=True
							)

	# Output is time of execution:
	# print("ERR: \n", proc.stderr)
	# print("OUT: \n", proc.stdout)

	try:

		time = float(proc.stdout)

		time = time # - Baseline.get()
		valid = True

		# update baseline
		# Baseline.update(time)

	except ValueError:
		print(proc.stdout)
		time = 0.0
		valid = False


	info = {"valid": valid, "reward": time, "baseline": Baseline.get()}
	return time, info
