#!/bin/bash -e
#SBATCH --mail-type=NONE          #Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-type=NONE
#SBATCH --mail-user=cdn16hqu@uea.ac.uk    # Where to send mail - change <username> to your  userid
#SBATCH --nodes=1       #limit to one node
#SBATCH -p  gpu-rtx6000-2       #Which queue to use
#SBATCH --qos=gpu-rtx
#SBATCH --gres=gpu:1              # Number of GPUs
#SBATCH --mem=41G               # memory
#SBATCH --array=1
#SBATCH --time=0-03:00:00            # time (DD-HH:MM)
#SBATCH --job-name=lstm_irm #Job name
#SBATCH -o ./results/lstm_irm_%j_%a.out       #Standard output log
#SBATCH -e ./results/lstm_irm_%j_%a.err       #Standard error log

# set up environment
module add matlab/2022b
echo $SLURM_ARRAY_TASK_ID

# run the application
echo Starting job $SLURM_JOB_ID
hostname

TMP=/gpfs/home/cdn16hqu/tcd_timit_basic/results/tcd_lstm_$SLURM_JOB_ID
mkdir -p $TMP

ENH=$TMP/enhanced/
IMG=$TMP/img_outputs/
mkdir -p $ENH $IMG

python3 -u main_clean_phase.py $TMP 64 40 'lps' 0.001

# Source folder path
# source_folder="./results"

# # Destination folder path
# destination_folder=$TMP

# # File names to move
# file1=lstm_irm_$SLURM_JOB_ID_$SLURM_ARRAY_TASK_ID.out 
# file2=lstm_irm_$SLURM_JOB_ID_$SLURM_ARRAY_TASK_ID.err  

# Check if the source files exist
# if [ -e "$source_folder/$file1" ] && [ -e "$source_folder/$file2" ]; then
#     # Move the files to the destination folder
#     mv "$source_folder/$file1" "$destination_folder/"
#     mv "$source_folder/$file2" "$destination_folder/"
#     echo "Files moved successfully."
# else
#     echo "One or both of the source files do not exist."
# fi

# Check if folder is not empty
# if [ "$(ls -A $ENH)" ]; then
#    echo "Output folder is not empty, starting MATLAB script..."
matlab -nodisplay -nosplash -nodesktop -nojvm -r "addpath('/gpfs/home/cdn16hqu/Matlab_eval'); addpath('${ENH}'); evaluateEnhancedAudioToExcel('/gpfs/home/cdn16hqu/NTCD_TIMIT/clean/test/', '${ENH}', 16000, 'evaluation.csv', ${SLURM_JOB_ID}); exit;"
# else
#    echo "Output folder is empty"
# fi

# Check the exit status of MATLAB
MATLAB_EXIT_STATUS=$?

# Check if MATLAB exited with an error status
if [ $MATLAB_EXIT_STATUS -ne 0 ]; then
    echo "MATLAB encountered an error and exited with status $MATLAB_EXIT_STATUS"
    exit $MATLAB_EXIT_STATUS  # Exit the bash script with the same error status
else
    echo "MATLAB executed successfully"
    exit 0  # Exit the bash script with a success status
fi




