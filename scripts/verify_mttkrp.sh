#!/bin/sh

# Script that automates running MTTKRP. Set the three params below then just run
# I recommend tee-ing script output to file

# -------------------

# Directory of tensor files
TENSOR_DIR=~/tensors/elements

# Strings to ignore if in filename
IGNORE_LIST="lbnl 5d"

# The command to run (without modes or input tensor specified)
BASE_COMMAND="./cpd128 -r 7 -k 3 -m 25 -p --device 1 --thread-cf 2 -c --stream-data -n 2"

# -------------------

echo "Command: $BASE_COMMAND"
echo

TENSORS=$(ls -Sr $TENSOR_DIR)
echo "Candidate tensors:"
echo $TENSORS
echo

# Iterate tensors in dir
for f in $TENSORS;
do
    # Ignore requested tensors
    for ignore in $IGNORE_LIST;
    do
        if [[ "$f" == *"$ignore"* ]]; then
            continue 2
        fi
    done

    ff="$TENSOR_DIR/$f"

    # Determine number of modes
    modes=$(head -n 1 $ff | wc -w)
    modes=$(($modes - 1))
    echo "Tensor $ff has $modes modes"

    # Iterate MTTKRP
    for (( i=0; i < modes; i++ ))
    do
        echo "MTTKRP mode $i"
        output=$($BASE_COMMAND -i $ff -t $i 2>&1 | grep -i -e "correct" -e "Error")
        echo "$output"
    done

    echo 
done
