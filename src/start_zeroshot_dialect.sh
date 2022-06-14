#!/bin/bash

python3.8 -u main_zeroshot_dialect.py \
--model classla/bcms-bertic \
--data bcms \
--batch_size 32 \
--random \
--type "$1" \
--device "$2"

python3.8 -u main_zeroshot_dialect.py \
--model classla/bcms-bertic \
--data bcms \
--batch_size 32 \
--type "$1" \
--device "$2"

python3.8 -u main_zeroshot_dialect.py \
--model classla/bcms-bertic \
--data bcms \
--batch_size 32 \
--mtl fixed \
--head masked \
--type "$1" \
--device "$2"

python3.8 -u main_zeroshot_dialect.py \
--model classla/bcms-bertic \
--data bcms \
--batch_size 32 \
--mtl uncertainty \
--head masked \
--type "$1" \
--device "$2"

python3.8 -u main_zeroshot_dialect.py \
--model classla/bcms-bertic \
--data bcms \
--batch_size 32 \
--mtl fixed \
--head cls \
--type "$1" \
--device "$2"

python3.8 -u main_zeroshot_dialect.py \
--model classla/bcms-bertic \
--data bcms \
--batch_size 32 \
--mtl uncertainty \
--head cls \
--type "$1" \
--device "$2"
