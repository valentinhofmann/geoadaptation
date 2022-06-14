#!/bin/bash

start=`date +%s`

for lr in 1e-05 3e-05 1e-04
do
	python3.8 -u main_geoadaptation.py \
	--batch_size 32 \
	--n_epochs 25 \
	--lr $lr \
	--model "$1" \
	--data "$2" \
	--device "$3"
done

end=`date +%s`

echo $((end-start)) >> "times/$2.txt"
