#!/bin/bash

# Link to cuda
export LD_LIBRARY_PATH=$LDLIBRARY_PATH:/USR/LOCAL/CUDA/LIB64

# Virtual envinrnoment
source ~/Env/python3/bin/activate


dir="/export/b02/fwu/MT/final_project/Beta-VAE-in-Machine-Translation/Fei/LanguageModel"
exe="$dir/main.py"
data="$dir/english"

echo $exe
echo $data

python3  $exe  --data $data --cuda




