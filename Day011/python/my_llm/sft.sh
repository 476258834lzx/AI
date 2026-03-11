#!/bin/bash

folder="sft_datas"
files=($(ls "$folder"))
num_files=${#files[@]}

for ((j=0; i<6; j++)); do
  for ((i=0; i<$num_files; i++)); do
    fn="${files[$i]}"

    if [ "$i" -gt -1 ]; then
      deepspeed sft.py --data_file "$folder/$fn" --ss $i
    fi
  done
done
