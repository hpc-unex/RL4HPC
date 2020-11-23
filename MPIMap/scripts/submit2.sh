#!/bin/sh

# Launch benchmarks for different kernels and configurations.


# Nodes
# node_nr="2 3 4 5 6 7 8 9 10 11 12"
node_nr="2"

# nets=("TCP" "IB")
nets=("IB")


for M in $node_nr; do                          # NODES

    for n in "${nets[@]}"; do                  # NET

        # Time to request
        t="02:00:00"

        echo -e " -J M"$M"_"$n" -t $t -p gpu --nodes=$M --wait-all-nodes=1 --exclusive ./rl_exec.sh $M $n"

        sbatch -J M"$M"_"$n" -t $t -p gpu --nodes=$M --wait-all-nodes=1 --exclusive  -x bc[03-18],bd[01-16],da10  ./rl_exec2.sh $M $n

    # block
    done

# network
done
