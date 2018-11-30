#!/bin/bash
# Corpus Cleaner
#Show location of the script
DIR="$( cd "$( dirname "$0" )" && pwd )"	#current location
echo "Script location: ${DIR}"

Data=/home/fei/Documents/JHU/18Fall/MT/FinalProject/Beta-VAE-in-Machine-Translation/Eng_Monoling_Data/masc_500k_texts/*

for dir in $Data; do
	if [ -d $dir ]; then # Folder exists
		folder=$dir/*
		echo "	${folder}"
		for sub_dir in $folder; do
			if [ -d $sub_dir ];then
				sub_folder=$sub_dir/*
		
				for file in $sub_folder; do 
					if [ -f $file ]; then	 # File exists
						echo "		Current file: ${file}"
						python3 CleanRaw.py $file
					fi
				done
			fi
		done
	fi
done

