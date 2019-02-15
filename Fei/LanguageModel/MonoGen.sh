#!/usr/bin/env bash

# DataFile=$1
# DataDir=$2
# echo "Data File:" $DataFile

DataDir=$1
echo "Data Dir:" $DataDir

# python3 SplitData.py $1
# mv train.txt $DataDir
# mv test.txt $DataDir
# mv valid.txt $DataDir
# mv $DataFile $DataDir 

python3 main.py --data $DataDir
python3 generate.py --data $DataDir
python3 CleanGenerated.py "generated.txt"

mv generated.txt $DataDir
mv model.pt $DataDir 

