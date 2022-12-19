#!/bin/bash
### install requirements for pstage3 baseline 
# JD update 12.19
# pip requirements
pip install torch==1.13
pip install datasets==2.7.1
pip install transformers==4.25.1
pip install tqdm
pip install pandas
pip install scikit-learn
pip install numpy
pip install glances
pip install omegaconf
pip install wandb==0.13.4
pip install pytorch-lightning==1.7.3
pip install pyYAML
pip install rich
pip install matplotlib
pip install black
pip install isort
pip install flake8
pip install pre-commit

# faiss install (if you want to)
pip install faiss-gpu
