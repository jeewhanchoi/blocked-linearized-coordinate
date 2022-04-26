#!/bin/sh

# Script that automates running MTTKRP. Set the three params below then just run
# I recommend tee-ing script output to file

# -------------------

# Directory of tensor files
TENSOR_DIR=~/tensors/elements

# Strings to ignore if in filename
IGNORE_LIST="lbnl patents amazon reddit 5d"

# The command to run (without mdoes or input tensor specified)
BASE_COMMAND="./cpd128 -r 32 -m 25 -p -k 10 --device 2 --thread-cf 11"

# -------------------

echo "Command: $BASE_COMMAND"
echo

# Iterate tensors in dir
for f in "$TENSOR_DIR"/*.tns;
do
    # Ignore requested tensors
    for ignore in $IGNORE_LIST;
    do
        if [[ "$f" == *"$ignore"* ]]; then
            continue 2
        fi
    done

    # Determine number of modes
    modes=$(head -n 1 $f | wc -w)
    modes=$(($modes - 1))
    echo "Tensor $f has $modes modes"

    # Iterate MTTKRP
    for (( i=0; i < modes; i++ ))
    do
        echo "MTTKRP mode $i"
        output=$($BASE_COMMAND -i $f -t $i 2>&1 | grep "Total time")
        echo "-> $output"
    done

    echo 
done
