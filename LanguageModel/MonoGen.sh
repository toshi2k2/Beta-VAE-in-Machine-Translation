#!/usr/bin/env bash
DataDir=$1
echo "Data stored at:" $DataDir
python3 main.py --data $DataDir
python3 generate.py --data $DataDir
python3 CleanGenerated.py "generated.txt"

