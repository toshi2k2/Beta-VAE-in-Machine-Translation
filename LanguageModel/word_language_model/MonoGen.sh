#!/usr/bin/env bash
DataDir="afrikaans"
echo "Data stored at:" $DataDir
python3 main.py --data $DataDir
python3 generate.py --data $DataDir
