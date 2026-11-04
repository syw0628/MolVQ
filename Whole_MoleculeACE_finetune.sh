#!/bin/bash
# MolculeACE

for dataset in $(printf '%s\n' ./Dataset/fine-tuning/preprocessed/MoleculeACE/*/ | sort) ; do
  echo "Starting ${dataset}..."
  dataset=$(basename "$dataset")

  python MoleculeACE_finetune.py \
      --dataset "${dataset}"  \
      --device 0 \
  done
done