#!/bin/bash

python generate.py data/phar_PARP1.posp gen_result/ checkpoints/fold0_epoch32.pth checkpoints/tokenizer_r_iso.pkl --filter --device cpu

