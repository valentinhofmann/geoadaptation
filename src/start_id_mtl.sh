#!/bin/bash

python3.8 -u main_id.py \
--batch_size 32 \
--n_epochs 10 \
--lr 3e-05 \
--generation 25 \
--model "$1" \
--data "$2" \
--mtl "$3" \
--head "$4" \
--device "$5"
