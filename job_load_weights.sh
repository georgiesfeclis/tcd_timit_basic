#!/bin/bash -e

#SBATCH --mail-type=NONE          #Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cdn16hqu@uea.ac.uk
#SBATCH --nodes=1
#SBATCH -p  gpu-rtx6000-2
#SBATCH --qos=gpu-rtx
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=0-03:00:00
#SBATCH --job-name=load_weights
#SBATCH -o /gpfs/home/cdn16hqu/tcd_timit_basic/results/tcd_lstm_16077984/res.out       #Standard output log
#SBATCH -e /gpfs/home/cdn16hqu/tcd_timit_basic/results/tcd_lstm_16077984/res.err      #Standard error log


python3 -u load_weights.py

