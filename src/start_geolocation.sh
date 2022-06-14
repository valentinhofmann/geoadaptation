#!/bin/bash

start=`date +%s`

for lr in 1e-05 3e-05 1e-04
do
	python3.8 -u main_geolocation.py \
	--batch_size 32 \
	--n_epochs 10 \
	--lr $lr \
	--model "$1" \
	--data "$2" \
	--generation "$3" \
	--geo_type "$4" \
	--device "$5"
done

end=`date +%s`

echo $((end-start)) >> "times_geolocation/$2_$3_$4.txt"
