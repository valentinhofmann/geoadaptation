#!/bin/bash

start=`date +%s`

python3.8 -u main_geolocation.py \
--batch_size 32 \
--n_epochs 10 \
--geo_type kmeans \
--model "$1" \
--data "$2" \
--mtl "$3" \
--head "$4" \
--generation "$5" \
--lr "$6" \
--device "$7"

end=`date +%s`

echo $((end-start)) >> "times_geolocation/$2_$3_$4_$5_kmeans.txt"
