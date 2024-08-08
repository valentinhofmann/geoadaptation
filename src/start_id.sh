#!/bin/bash

for lr in 3e-05 1e-05 1e-04
do
	python3.8 -u main_id.py \
	--batch_size 32 \
	--n_epochs 10 \
	--generation 25 \
	--lr $lr \
	--model "$1" \
	--data "$2" \
	--device "$3"
done
