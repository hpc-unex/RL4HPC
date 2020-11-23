#!/bin/sh
##############################################################################
#   .,**ooooo***.       ',***********      ',************          *****.
#  .ooooooooooooo       '*ooooooooooo      .ooooooooooooo         'oooooo.
#  *ooo.''''',ooo       '*o*.''''''''       '''',ooo.''''        '*oo,.*oo.
# '*ooo       '''       '*oo*,,,,,,,,           .ooo'           '*oo,  .oo*'
# '*ooo                 '*oooooooooo*           .ooo'           ,oo,    ooo*'
# '*ooo      ',**'      '*o*.''''''''           .ooo'          .ooo*****oooo*
#  ,ooo*,,,,,*ooo'      '*oo*,,,,,,,,           .ooo'         .oooooooooooooo
#  '*oooooooooooo       '*ooooooooooo'          .ooo'         oooo,''''''.*oo.
#    '.,,,,,,,,.'        .,.........,           '.,,          ,.,.        '...
#
#                          CETA-CIEMAT GPU CLUSTER
#                   ***************************************
#
#                        Example 02: MPI + CUDA example
#
##############################################################################

#SBATCH --mem=20000M

# PARAMETERS:
#   $1 Number of nodes.
#   $2 Block size
# Description:
#   Creates a directories structure for executing FuPerMod FPMs model generation and partition.
#   - 1) It creates a Hostfile (for MPI), RankFile (for MPI and ConfFile (for FuPerMod)
#        from the nodes the system reserves. It uses three 8-core/2-GPUs and a 12-core/2-GPUs for
#        generating the configuration file for FuPerMod, as heterogeneous as possible.
#
#        a) Execution of benchmarks for obtaining the FPMs of each process (depending of the kernel: MxM / W2D)
#        b) Partition of the matrix using the FPMs (inluding a 1D partitioning)
#

# LIMITATIONS:
#   2) Only one configuration for each node number (it can change if the nodes received from the
#      system are of different types).
#   3) We expect a number of nodes of two types: 8-core and 12-core, both with 2 processors and 2 GPUs.
#

# TBD:
#   - cuda_gemm.cpp: There is some bug in this file, line 52 ("out of memory") when P>> or NREPS>>
#   - Test more use cases ...





t_init=`date +%s`




# Modules and libraries

# OpenMPI
# module load openmpi_gcc
module load openmpi_gcc_cuda


# MKL: do not know if works. Not needed.
module load intel/2016_update3
export LD_LIBRARY_PATH=/opt/intel/2016_update3/mkl/lib/intel64/:$LD_LIBRARY_PATH


# CUDA
module load cuda/8.0.61



# PARAMETERS: Nodes, kernels, modes, block numbers, etc.
M=$1
net=$2
m=$3


echo -e "PARAMTERS:\n   M: $M\n   Net: $net\n   m: $m\n"

numProcs=8



# Global variables: Directories.
ROOT_FOLDER=/home/jarico/RL4HPC
DATA_FOLDER=$ROOT_FOLDER/data
OUTPUT_FOLDER=$ROOT_FOLDER/output
BIN_FOLDER=$ROOT_FOLDER/bin

MPIEXEC=/opt/openmpi_gcc/2.0.2/bin/mpirun

# Copy of previous output file
mv -f $OUTPUT_FOLDER/output.txt $OUTPUT_FOLDER/output_`date +"%Y-%m-%d_%H-%M"`.txt


# Write the names of the nodes allocated by SLURM to a file (ordered)
scontrol show hostname ${SLURM_JOB_NODELIST} | sort > $DATA_FOLDER/hosts


# Parameters to Open MPI are not useful. FPM are independent from network. Actually, not independet from the kernel,
#  but now MxM is used for generating models for all kernels.
if [ "$net" == "TCP" ]; then
  COMMS=" tcp,self,sm "
else
  COMMS=" openib,self,sm "
fi

OMPIALG=" --mca coll_tuned_priority 100 --mca coll_tuned_use_dynamic_rules 1 --mca coll_tuned_bcast_algorithm 6 "


echo "$m" > $DATA_FOLDER/IMB_lenghts
IMB_OPTIONS="Bcast -npmin $numProcs -root_shift off -imb_barrier on -time 20 -iter_policy off -off_cache -1 -iter 10000 -msglen $DATA_FOLDER/IMB_lenghts"


# echo -e "/opt/openmpi_gcc/2.0.2/bin/mpirun -n $numProcs --hetero-nodes --hostfile $HOSTFILE --rankfile $RANKFILE --mca btl $COMMS $OMPIALG  --bind-to core  -report-bindings --display-map  -nooversubscribe $BIN_FOLDER/IMB_MPI1 $IMB_OPTIONS"


python3 $ROOT_FOLDER/code/REINFORCE_colls.py "$M" "$numProcs" "$net" "$m" "$MPIEXEC" "--mca btl $COMMS" "$OMPIALG --bind-to core -report-bindings --display-map -nooversubscribe" "$BIN_FOLDER/IMB-MPI1" "$IMB_OPTIONS" "$DATA_FOLDER/graph_file.txt" "$DATA_FOLDER" "$OUTPUT_FOLDER/output.txt"

# python3 $ROOT_FOLDER/code/rl_gettime.py "$M" "$numProcs" "$net" "$m" "$MPIEXEC" "--mca btl $COMMS" "$OMPIALG" "$BIN_FOLDER/IMB-MPI1" "$IMB_OPTIONS"

# OLD: /opt/openmpi_gcc/2.0.2/bin/mpirun -n $numProcs --hetero-nodes --hostfile $HOSTFILE --rankfile $RANKFILE --mca btl $COMMS $OMPIALG  --bind-to core  -report-bindings --display-map  -nooversubscribe  $BIN_FOLDER/IMB_MPI1 $IMB_OPTIONS
# /opt/openmpi_gcc/2.0.2/bin/mpirun -n $numProcs  --mca btl $COMMS $OMPIALG  --bind-to core  -report-bindings --display-map  -nooversubscribe  $BIN_FOLDER/IMB-MPI1 $IMB_OPTIONS


if [ "$?" -eq 0 ]; then
    echo "OK"
else
    echo "FAIL"
fi



exit 0


# Global variables: Directories.
FUPERMOD_FOLDER=/home/jarico/fupermod-latest
ROOT_FOLDER=/home/jarico/fupermod-latest/ccgrid_tests

mkdir -p $ROOT_FOLDER/Ciemat

KERNEL_FOLDER=$ROOT_FOLDER/Ciemat/"$kernel"
NODE_FOLDER=$KERNEL_FOLDER/K"$K"/M"$numNodes"
config_name="$kernel"_"$mode"_"$net"_"$blocks"
CONFIG_FOLDER=$NODE_FOLDER/"$config_name"
FPM_FOLDER=$CONFIG_FOLDER/fpm

echo -e "KERNEL: "
echo -e $KERNEL_FOLDER
echo -e "NODE: "
echo -e $NODE_FOLDER
echo -e "CONFIG: "
echo -e $CONFIG_FOLDER
echo -e "FPM: "
echo -e $FPM_FOLDER


# Binaries and libraries for each kernel/mode/kb/... are in BIN/LIB_FOLDER
# (Name is e.g.: mxm_2d, or builder)
BIN_FOLDER=$ROOT_FOLDER/bin
LIB_FOLDER=$ROOT_FOLDER/lib


# Suboptions for the FPM construction:
#   k = kb (size of the block) - Take in as parameter
#   N = size in blocks - Take in as parameter
#   i = iterations for constructing models (FPM only, not for the benchmark) - Default=1
subopts=",k="$K",N="$blocks",i=1"
echo -e "Sub-options: "
echo -e $subopts


# Use only 12-core nodes
# Two patterns
# SUBOPTS
#patt_12=("NODE 0 0-5 cpu OMP_NUM_THREADS=6"$subopts"
#NODE 1 6-9 cpu OMP_NUM_THREADS=4"$subopts"
#NODE 2 10 gpu id=0"$subopts"
#NODE 3 11 gpu id=1"$subopts"" "NODE 0 0-5 cpu OMP_NUM_THREADS=6"$subopts"
#NODE 1 6-9 cpu OMP_NUM_THREADS=4"$subopts"
#NODE 2 10 gpu id=0"$subopts"
#NODE 3 11 gpu id=1"$subopts"")

#PATT_NRS=2


# SECOND PATTERN: less nr of processes
# Use only 12-core nodes
# Two patterns
# SUBOPTS
patt_12=("NODE 0 0-10 cpu OMP_NUM_THREADS=11"$subopts"
NODE 1 11 gpu id=0"$subopts"" "NODE 0 0-9 cpu OMP_NUM_THREADS=10"$subopts"
NODE 1 10 gpu id=0"$subopts"
NODE 2 11 gpu id=1"$subopts"")

PATT_NRS=2




# Main

# Creating the folder structure
mkdir -p $KERNEL_FOLDER
mkdir -p $KERNEL_FOLDER/K"$K"
mkdir -p $NODE_FOLDER

rm -rf $CONFIG_FOLDER

mkdir -p $CONFIG_FOLDER
mkdir -p $FPM_FOLDER



# Hosts file
HOSTFILE=$FPM_FOLDER/hosts
HOSTFILE_TMP=$FPM_FOLDER/tmp_hosts


# Write the names of the nodes allocated by SLURM to a file (ordered)
scontrol show hostname ${SLURM_JOB_NODELIST} | sort > $HOSTFILE_TMP

# Configuration pattern for the node types
CFILE=$FPM_FOLDER/M"$numNodes".conf

# Header of CFILE: from this file we generate the RANKFILE and the CONFFILE
echo -e "# Configuration: M"$numNodes  > $CFILE


# Number of procs
i=0

# Pattern to apply
patt_8_nr=0
patt_12_nr=0

# Number of processes is calculated from the configuration applied.
numProcs=0

# Patterns from hostfile to a preliminar configuration file in CFILE
for line in $(cat $HOSTFILE_TMP)
do

    core_nr=0

    case "$line" in
        "bd01" ) core_nr=8;;
        "bd02" ) core_nr=8;;
        "bd03" ) core_nr=8;;
        "bd04" ) core_nr=8;;
        "bd05" ) core_nr=8;;
        "bd06" ) core_nr=8;;
        "bd07" ) core_nr=8;;
        "bd08" ) core_nr=8;;
        "bd09" ) core_nr=8;;
        "bd10" ) core_nr=8;;
        #  "bd11" ) core_nr=8;; # No GPU connected
        #  "bd12" ) core_nr=8;; # No GPU connected
        "bd13" ) core_nr=8;;
        "bd14" ) core_nr=8;;
        "bd15" ) core_nr=8;;
        "bd16" ) core_nr=8;;

        "bd17" ) core_nr=12;;
        "bd18" ) core_nr=12;;
        "bd19" ) core_nr=12;;
        "bd20" ) core_nr=12;;
        "bd21" ) core_nr=12;;
        "bd22" ) core_nr=12;;
        "bd23" ) core_nr=12;;
        "bd24" ) core_nr=12;;
        "bd25" ) core_nr=12;;
        "bd26" ) core_nr=12;;
        "bd27" ) core_nr=12;;
        "bd28" ) core_nr=12;;
        "bd29" ) core_nr=12;;
        "bd30" ) core_nr=12;;
        "bd31" ) core_nr=12;;
        "bd32" ) core_nr=12;;
    esac

    # Write the hostfile name and the number of slots.
    if [ "$core_nr" -eq 8 ]; then
        echo $line" slots=8"  >> $HOSTFILE
    else
        echo $line" slots=12" >> $HOSTFILE
    fi


    # List of nodes
    node_name[$i]="$line"
    let "i += 1"

    # Write the .conf file
    if [ "$core_nr" -eq "8" ]; then
        c_line=${patt_8[$patt_8_nr]//NODE/$line}
        let "patt_8_nr = (patt_8_nr + 1) % $PATT_NRS"
    else
        c_line=${patt_12[$patt_12_nr]//NODE/$line}
        let "patt_12_nr = (patt_12_nr + 1) % $PATT_NRS"
    fi
    # Write the line.
    echo -e "$c_line" >> $CFILE

done


# Rank file
# Create the files in the correct folder
RANKFILE=$FPM_FOLDER/rnk_file_conf
CONFFILE=$FPM_FOLDER/conf_file

# Header of RANKFILE for MPI
echo -e "# MPI RANKFILE: $RANKFILE"  > $RANKFILE

# Header of CONFFILE for FuPerMod
echo -e "# FuPerMod CONFFILE: $CONFFILE"  > $CONFFILE


# From each line of the Mx.conf file we create a line in each one of
#  the CONFFILE and RANKFILE files, with the correct format.
rank_nr=0
while IFS='' read -r line || [[ -n "$line" ]]; do

    new_line=( $line )

    # Skip or copy the comment lines
    carac=${line:0:1}
    if [ "$carac" == "#" ]; then
        echo $line >> $RANKFILE
    elif [ ! -z "$line" ]
    then

        conf_line="${new_line[0]} ${new_line[1]} ${new_line[2]} ${new_line[3]} ${new_line[4]}"
        rank_line="rank $rank_nr=${new_line[0]} slot=${new_line[2]}"

        let "rank_nr += 1"

        # Copy to files
        echo -e $conf_line >> $CONFFILE
        echo -e $rank_line >> $RANKFILE

    fi

done < $CFILE

numProcs=$rank_nr



# File with the configuration options (Not needed by now)
#COPTIONS=$KNL_FOLDER/"$mode"/"$name"/conf_opt
#echo -e "#Configutation options for $name" > $COPTIONS
#echo -e $kernel" #Kernel"     >> $COPTIONS
#echo -e $mode" #Mode"         >> $COPTIONS
#echo -e $net" #Network"       >> $COPTIONS
#echo -e $blocks" #Blocks (N)" >> $COPTIONS
#echo -e $numNodes" #M"        >> $COPTIONS
#echo -e $numProcs" #P"        >> $COPTIONS
#echo -e "5 #NREPS"            >> $COPTIONS




################################################################################################################
# Directories and configuration/Rankfile files have been created. Now execute the benchmarks in 3 steps:
#
#    1. Obtain the FPMs of the processes
#    2. Partitioning the data space
#    3. Run the benchmarks for the communication patterns (p2p, ring and bcast)
#
echo -e "-----------------------------------------------------"
echo -e "    Configuration:    $name"




#####  1.  FPMs
#
# Policy: --map-by core   (Not valid because we provide with a rankfile)
# Bcast algorithm (Not used for FPMs, only computation):
#  1 -> Basic linear (isend_init/start)
#  6 -> Binomial

cd $FPM_FOLDER
echo -e "  --->>  Calculating FPMs ... \n\n"


# Parameters to Open MPI are not useful. FPM are independent from network. Actually, not independet from the kernel,
#  but now MxM is used for generating models for all kernels.
if [ "$net" == "TCP" ]; then
  COMMS=" tcp,self,sm "
else
  COMMS=" openib,self,sm "
fi


# cp $CONFFILE $FPM_FOLDER

OMPIALG=" --mca coll_tuned_priority 100 --mca coll_tuned_use_dynamic_rules 1 --mca coll_tuned_bcast_algorithm 1 "

# LIBS=" $BIN_FOLDER/bin/builder_"$kernel"_"$mode"_"$K" -l $BIN_FOLDER/lib/libwave_1d_"$kernel"_"$mode"_"$K".so "
if [ "$kernel" == "MxM" ]; then
  echo -e " MxM Kernel "
  LIBS=" $FUPERMOD_FOLDER/bin/builder -l $FUPERMOD_FOLDER/lib/libmxm_1d.so "
  LIBS_OPTS=" -L16 -U1024 -s64 -i0.98 -r1000 "
else
  echo -e " W2D Kernel "
  LIBS=" $FUPERMOD_FOLDER/bin/builder -l $FUPERMOD_FOLDER/lib/libwave_2d.so "
  LIBS_OPTS=" -L16 -U512 -s32 -i0.98 -r10 -c4 -m1 "
fi

echo -e "/opt/openmpi_gcc/2.0.2/bin/mpirun -n $numProcs --hetero-nodes --hostfile $HOSTFILE --rankfile $RANKFILE --mca btl $COMMS $OMPIALG  --bind-to core  -report-bindings --display-map  -nooversubscribe $LIBS $LIBS_OPTS"

/opt/openmpi_gcc/2.0.2/bin/mpirun -n $numProcs --hetero-nodes --hostfile $HOSTFILE --rankfile $RANKFILE --mca btl $COMMS $OMPIALG  --bind-to core  -report-bindings --display-map  -nooversubscribe  $LIBS $LIBS_OPTS


if [ "$?" -eq 0 ]; then
    echo "OK"
else
    echo "FAIL"
fi





#####  2.  Partitioning
#
# Depends on k and N
# We take k as a constant (block size)
# Values of N and D are calculated

echo -e "  --->>  Calculating Partitions ... \n\n"

# D is the number of blocks in the matrix
#echo $(bc <<< $blocks * $blocks)
#let "D=$(bc <<< $blocks * $blocks)"

let "D = $blocks * $blocks"
# N is the number of elements in a row or a column
let "N = $blocks * $K"

echo -e "$name:  b = $blocks,  D = $D,  N = $N  K = $K"

# Partition of a matrix 256x256 example:
$FUPERMOD_FOLDER/bin/partitioner -l $FUPERMOD_FOLDER/lib/libmxm_1d.so -a3 -D $D  -o k="$K" -p $FPM_FOLDER/part.dist
$FUPERMOD_FOLDER/bin/1dto2d_dist -k "$K" -N $N -q $FPM_FOLDER/part.2dist -p $FPM_FOLDER/part.dist

# Generate also a 1D partition
python $FUPERMOD_FOLDER/bin/part_1d.py "$blocks" "$K" $FPM_FOLDER/part.dist $HOSTFILE $FPM_FOLDER/part.1dist


# cd $FPM_FOLDER

if [ "$?" -eq 0 ]; then
    echo "OK"
else
    echo "FAIL"
fi




t_end=`date +%s`
let total=$t_end-$t_init
let total_m=$total/60
echo "Wallclock time:  -$total- secs  -$total_m- mins"


exit 0
