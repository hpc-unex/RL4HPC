#!/bin/sh

# Launch benchmarks for different kernels and configurations.


# Nodes
# node_nr="2 3 4 5 6 7 8 9 10 11 12"
node_nr="2"

# nets=("TCP" "IB")
nets=("IB")

# sizes="128 256 512 1024"
sizes="524288"


for i in $node_nr; do                          # NODES

    for n in "${nets[@]}"; do                  # NET

        for sz in $sizes; do                   # SIZES (N)

                # Time to request
            t="05:30:00"

            echo -e " -J M"$i"_"$n"_"$sz" -t $t -p gpu --nodes=$i --wait-all-nodes=1 --exclusive ./rl_exec.sh $i $n $sz"

            sbatch -J M"$i"_"$n"_"$sz" -t $t -p gpu --nodes=$i --wait-all-nodes=1 --exclusive  -x bc[03-18],bd[01-16],da10  ./rl_exec.sh $i $n $sz

        # sizes
        done

    # block
    done

# network
done
