#!/bin/bash

python3.8 -u main_zeroshot_geolocation.py \
--model classla/bcms-bertic \
--data bcms \
--batch_size 32 \
--random \
--location "$1" \
--device "$2"

python3.8 -u main_zeroshot_geolocation.py \
--model classla/bcms-bertic \
--data bcms \
--batch_size 32 \
--pretrained \
--location "$1" \
--device "$2"

python3.8 -u main_zeroshot_geolocation.py \
--model classla/bcms-bertic \
--data bcms \
--batch_size 32 \
--location "$1" \
--device "$2"

python3.8 -u main_zeroshot_geolocation.py \
--model classla/bcms-bertic \
--data bcms \
--batch_size 32 \
--mtl fixed \
--head masked \
--location "$1" \
--device "$2"

python3.8 -u main_zeroshot_geolocation.py \
--model classla/bcms-bertic \
--data bcms \
--batch_size 32 \
--mtl uncertainty \
--head masked \
--location "$1" \
--device "$2"

python3.8 -u main_zeroshot_geolocation.py \
--model classla/bcms-bertic \
--data bcms \
--batch_size 32 \
--mtl uncertainty \
--head cls \
--location "$1" \
--device "$2"
