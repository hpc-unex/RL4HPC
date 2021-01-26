import numpy as np
import subprocess


def state_to_list(state, P):
	
	l = list()
	
	# print("Graph for: ", P, " processes.")
	# print(state, flush=True)
	
	for s in range(P):
		for r in range(P):
			stage = state[s,r]
			if stage != 0:
				# print(s, r, stage, flush=True)
				l.append((s, r, stage))

	return l



# TBD:
def generate_hosts_files(params):
	
	hosts_cur_name = params["hosts_list"]
	
	hosts_list = list()
	
	# Write the host file
	hosts_f_name  = params["hosts_file"]
	hosts_file = open(hosts_f_name,"w+")
	with open(hosts_cur_name, 'r') as h_file:
		for host in h_file:
			hosts_file.write(host.rstrip() + " " + "slots=24\n")
			hosts_list.append(host.rstrip())
	h_file.close()
	
	hosts_file.close()
	

	# Write the rank file (TODO)
	rank_f_name = params["rank_file"]
	rank_file  = open(rank_f_name,"w+")

#for p in range(P):
#    'rank {}={} slot=0-5'.format(p, host, "0-5")
rank_file.write('rank {}={} slot={}\n'.format(0, hosts_list[0], "0-5"))
rank_file.write('rank {}={} slot={}\n'.format(1, hosts_list[1], "0-5"))
rank_file.write('rank {}={} slot={}\n'.format(2, hosts_list[0], "6-11"))
rank_file.write('rank {}={} slot={}\n'.format(3, hosts_list[1], "6-11"))
rank_file.write('rank {}={} slot={}\n'.format(4, hosts_list[0], "12-17"))
rank_file.write('rank {}={} slot={}\n'.format(5, hosts_list[1], "12-17"))
rank_file.write('rank {}={} slot={}\n'.format(6, hosts_list[0], "18-23"))
rank_file.write('rank {}={} slot={}\n'.format(7, hosts_list[1], "18-23"))

rank_file.close()

	
	# Names for files
	# params["hosts_file"] = hosts_f_name
	# params["rank_file"] = rank_f_name
	
	
	return





def get_reward_mpi(state, M, params, iter=0):
	
	t = 0.0
	
	params_output = config["Output"]
	generate_hosts_files(params_output)
	
	graph_file  = params['graph_file']
	output_file = params['output_file']

	l_graph = state_to_list(state, params["P"])
	l_graph.sort(key=lambda x:(x[2], x[1], x[0])) # Sort by: Stage - Dst - Src

	# Graph file containing the description of the graog of communications
	g_file = open(graph_file, 'w');
	g_file.write('\n'.join('%d %d %d' % (s,r,stage) for (s,r,stage) in l_graph) )
	g_file.close()

	# Output file contains output messages
	o_file = open(output_file, 'a')
	o_file.write("\nOutput File: " + str(iter) + "\n")
	o_file.write('\n'.join('%d %d %d' % (s,r,stage) for (s,r,stage) in l_graph) )


	exec_command = params["exec_command"]
	# print(exec_command)
	
	# -n $numProcs  --mca btl $COMMS $OMPIALG  --bind-to core  -report-bindings --display-map  -nooversubscribe  $BIN_FOLDER/IMB-MPI1 $IMB_OPTIONS
	proc = subprocess.run(exec_command,
						  stdout=subprocess.PIPE,
						  stderr=subprocess.PIPE,
						  shell=False,
						  universal_newlines=True)
							# check=True,
	
	# (output, err) = proc.communicate()
	output = proc.stdout
	err = proc.stderr
				
	if (err != None):
		print(err)


	data = [0.0, 0.0, 0.0, 0.0, 0.0]

	for line in output.splitlines():
		if len(line) > 0:
			w = line.lstrip(" \t")
			# print(w, flush=True)
			if (w[0] != '#'):
				# print(w, flush=True)
				try:
					data = [float(i) for i in w.split()]
				except:
					None  # print(w.split(), flush=True)

	# proc.wait()

	# Format of IMB return
	# data = [m , reps, t_min, t_max, t_avg]

	t = data[4] # AVG

	P = int(params['P'])
	o_file.write("\nREWARD: " + str(t/P) + "\n")
	# o_file.write(output)
	o_file.write("IMB: " + str(data).strip('[]') + "\n")
	o_file.close()
			
	return t/P

