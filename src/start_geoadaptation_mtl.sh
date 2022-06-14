#!/bin/bash

start=`date +%s`

python3.8 -u main_geoadaptation.py \
--batch_size 32 \
--n_epochs 25 \
--lr 3e-05 \
--model "$1" \
--data "$2" \
--mtl "$3" \
--head "$4" \
--device "$5"

end=`date +%s`

echo $((end-start)) >> "times/$2_$3_$4.txt"
