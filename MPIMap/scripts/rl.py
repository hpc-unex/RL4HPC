import os
from subprocess import PIPE, Popen, call
import numpy as np
from decimal import Decimal
from numpy import sort, floor

import sys
import subprocess

from mpi4py import MPI


def launch_mpi (params):

    t = 0.0

    exec_command = params["exec_command"]

    # exec_command = "/home/jarico/RL4HPC/bin/example"

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

    print(output)

    return t





#  MAIN


params = dict()


M   = int(sys.argv[1])

P   = int(sys.argv[2])
net = sys.argv[3]
m   = int(sys.argv[4])

IMB     = sys.argv[5]
IMBOPT  = sys.argv[6]

graph_file  = sys.argv[7]
hosts_file  = sys.argv[8]
output_file = sys.argv[9]

# print('Parameters: (M, P, net, m): ', M, P, net, m)

params["P"] = P
params["M"] = M
params["m"] = m
params["net"] = net

params["graph_file"]  = graph_file
params["output_file"] = output_file
params["hosts_file"]  = hosts_file
params["rank_file"]   = ""


# generate_hosts_files(params)


OPTIONS = IMB + " " + IMBOPT
EXEC = OPTIONS.split()
params["exec_command"] = EXEC



MPI.Init()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print('My rank is ',rank)
MPI.Finalize()



print("-----------------")
print(params, flush=True)


launch_mpi(params)
