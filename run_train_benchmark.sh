#!/bin/bash

mkdir -p logs

source .venv/bin/activate

LOSS_TYPE=torch LIGER_MODEL=no BATCH_SIZE=1 python3 train_llama.py
LOSS_TYPE=torch LIGER_MODEL=no BATCH_SIZE=2 python3 train_llama.py
LOSS_TYPE=torch LIGER_MODEL=no BATCH_SIZE=4 python3 train_llama.py

LOSS_TYPE=torch LIGER_MODEL=yes BATCH_SIZE=1 python3 train_llama.py
LOSS_TYPE=torch LIGER_MODEL=yes BATCH_SIZE=2 python3 train_llama.py
LOSS_TYPE=torch LIGER_MODEL=yes BATCH_SIZE=4 python3 train_llama.py
LOSS_TYPE=torch LIGER_MODEL=yes BATCH_SIZE=8 python3 train_llama.py
LOSS_TYPE=torch LIGER_MODEL=yes BATCH_SIZE=12 python3 train_llama.py

LOSS_TYPE=liger LIGER_MODEL=no BATCH_SIZE=1 python3 train_llama.py
LOSS_TYPE=liger LIGER_MODEL=no BATCH_SIZE=2 python3 train_llama.py
LOSS_TYPE=liger LIGER_MODEL=no BATCH_SIZE=4 python3 train_llama.py
LOSS_TYPE=liger LIGER_MODEL=no BATCH_SIZE=8 python3 train_llama.py

LOSS_TYPE=liger LIGER_MODEL=yes BATCH_SIZE=1 python3 train_llama.py
LOSS_TYPE=liger LIGER_MODEL=yes BATCH_SIZE=2 python3 train_llama.py
LOSS_TYPE=liger LIGER_MODEL=yes BATCH_SIZE=4 python3 train_llama.py
LOSS_TYPE=liger LIGER_MODEL=yes BATCH_SIZE=8 python3 train_llama.py
LOSS_TYPE=liger LIGER_MODEL=yes BATCH_SIZE=12 python3 train_llama.py

LOSS_TYPE=ligerfusedlinear LIGER_MODEL=no BATCH_SIZE=1 python3 train_llama.py
LOSS_TYPE=ligerfusedlinear LIGER_MODEL=no BATCH_SIZE=2 python3 train_llama.py
LOSS_TYPE=ligerfusedlinear LIGER_MODEL=no BATCH_SIZE=4 python3 train_llama.py
LOSS_TYPE=ligerfusedlinear LIGER_MODEL=no BATCH_SIZE=8 python3 train_llama.py
LOSS_TYPE=ligerfusedlinear LIGER_MODEL=no BATCH_SIZE=12 python3 train_llama.py

LOSS_TYPE=ligerfusedlinear LIGER_MODEL=yes BATCH_SIZE=1 python3 train_llama.py
LOSS_TYPE=ligerfusedlinear LIGER_MODEL=yes BATCH_SIZE=2 python3 train_llama.py
LOSS_TYPE=ligerfusedlinear LIGER_MODEL=yes BATCH_SIZE=4 python3 train_llama.py
LOSS_TYPE=ligerfusedlinear LIGER_MODEL=yes BATCH_SIZE=8 python3 train_llama.py
LOSS_TYPE=ligerfusedlinear LIGER_MODEL=yes BATCH_SIZE=12 python3 train_llama.py

LOSS_TYPE=ligerfusedlinear LIGER_MODEL=yes BATCH_SIZE=1 TILED_MLP=yes python3 train_llama.py
LOSS_TYPE=ligerfusedlinear LIGER_MODEL=yes BATCH_SIZE=2 TILED_MLP=yes python3 train_llama.py
LOSS_TYPE=ligerfusedlinear LIGER_MODEL=yes BATCH_SIZE=4 TILED_MLP=yes python3 train_llama.py
LOSS_TYPE=ligerfusedlinear LIGER_MODEL=yes BATCH_SIZE=8 TILED_MLP=yes python3 train_llama.py
LOSS_TYPE=ligerfusedlinear LIGER_MODEL=yes BATCH_SIZE=12 TILED_MLP=yes python3 train_llama.py
LOSS_TYPE=ligerfusedlinear LIGER_MODEL=yes BATCH_SIZE=16 TILED_MLP=yes python3 train_llama.py
LOSS_TYPE=ligerfusedlinear LIGER_MODEL=yes BATCH_SIZE=24 TILED_MLP=yes python3 train_llama.py
LOSS_TYPE=ligerfusedlinear LIGER_MODEL=yes BATCH_SIZE=32 TILED_MLP=yes python3 train_llama.py
