#!/bin/bash
CONFIGS=("base_config")

config_length=${#CONFIGS[@]}

for (( i=0; i<${config_length}; i++ ));
do
    echo ${CONFIGS[$i]}
    python3 main.py \
        --config ${CONFIGS[$i]}
done