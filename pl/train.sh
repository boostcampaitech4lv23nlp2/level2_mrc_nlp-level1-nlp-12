#!/bin/bash
CONFIGS=("sweep1" "sweep2" "sweep3" "sweep4" "sweep5" "sweep6")

config_length=${#CONFIGS[@]}

for (( i=0; i<${config_length}; i++ ));
do
    echo ${CONFIGS[$i]}
    python3 main.py \
        --config ${CONFIGS[$i]}
done