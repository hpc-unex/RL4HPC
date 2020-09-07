import os
from subprocess import PIPE, Popen, call
import numpy as np
from decimal import Decimal
from numpy import sort, floor
import subprocess

import sys






def call_mpi(M, P, m, EXEC):

    exec_command = EXEC

    print(exec_command)

    # -n $numProcs  --mca btl $COMMS $OMPIALG  --bind-to core  -report-bindings --display-map  -nooversubscribe  $BIN_FOLDER/IMB-MPI1 $IMB_OPTIONS
    proc = subprocess.run(   exec_command,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             shell=False,
                             check=True,
                             universal_newlines=True)
    # (output, err) = proc.communicate()
    output = proc.stdout
    err = proc.stderr

    if (err != None):
        print(err)

    print(output)

    # proc.wait()

    time = 99 # Por ahora no conectado.

    return time


# MAIN

print("Get Time of Collective Operation")

M   = int(sys.argv[1])
P   = int(sys.argv[2])
net = sys.argv[3]
m   = int(sys.argv[4])

MPIEXEC = sys.argv[5]
COMMS   = sys.argv[6]
OMPIALG = sys.argv[7]
IMB     = sys.argv[8]
IMBOPT  = sys.argv[9]

print('Parameters: (M, P, net, m): ', M, P, net, m)
# print("Other options:")
# print(COMMS)
# print(OMPIALG)
# print(IMB)
# print(IMBOPT)

OPTIONS = MPIEXEC + " -n " + str(P) + " " + COMMS + " " + OMPIALG + " " + IMB + " " + IMBOPT
EXEC = OPTIONS.split()

print("-----------------")
print(EXEC)

t = call_mpi(M, P, m, EXEC)
print("Time: ", t)
