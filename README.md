# Geoadaptation of Language Models 🌍

This repository contains the code for the TACL 2024 paper [Geographic Adaptation of Pretrained Language Models](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00652/120648/Geographic-Adaptation-of-Pretrained-Language).

# Usage 

The main code for conducting geoadaptation can be found in `src/main_geoadaptation.py`.

To replicate the geoadaptation experiments from the paper, use the scripts `src/start_geoadaptation.sh` and `src/start_geoadaptation_mtl.sh`.

To replicate the experiments on fine-tuned geolocation prediction from the paper, use the scripts `src/start_geolocation.sh` and `src/start_geolocation_mtl.sh`.

To replicate the experiments on zero-shot geolocation prediction from the paper, use the script `src/start_zeroshot_geolocation.sh`.

To replicate the experiments on zero-shot dialect feature prediction from the paper, use the script `src/start_zeroshot_dialect.sh`.

# Citation

If you use the code in this repository, please cite the following paper:

```
@article{hofmann2024geoadaptation,
    title = {Geographic Adaptation of Pretrained Language Models},
    author = {Hofmann, Valentin and Glavaš, Goran and Ljubešić, Nikola and Pierrehumbert, Janet and Schütze, Hinrich},
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {12},
    pages = {411-431},
    year = {2024}
}
```
