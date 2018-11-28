#!/usr/bin/env bash
DataDir="/home/fei/Documents/JHU/18Fall/MT/FinalProject/Beta-VAE-in-Machine-Translation/LanguageModel/afrikaans"
echo "Data stored at:" $DataDir
python3 main.py --data $DataDir
python3 generate.py --data $DataDir
