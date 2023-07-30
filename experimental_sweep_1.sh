#!/bin/bash
#SBATCH --job-name=sweep_1.1        # Name of the job
#SBATCH --output=output_%A_%a.txt       # Output file for each job array task
#SBATCH --error=error_%A_%a.txt         # Error file for each job array task
#SBATCH --array=0-63                   # Number of tasks in the array (e.g., 10 jobs)
#SBATCH --gres=gpu:1                   # Number of GPUs to allocate per job (1 GPU per job)
#SBATCH --time=02:00:00                # Maximum runtime for each job (hh:mm:ss)
#SBATCH --ntasks=1         # number of processor cores (i.e. tasks)
#SBATCH --nodes=1          # number of nodes
#SBATCH --mem-per-cpu=64G           # Memory required per job

# Load any required modules (if needed)
module load cuda/11.8

# Activate the virtual environment (if needed)
# source /path/to/your/virtualenv/bin/activate

# Change to the working directory where your code is located
# cd /path/to/your/code_directory

# Run your Python script with the GPU device specified (assuming your script is named 'your_script.py')
python experimental_sweep_1.py --id $SLURM_ARRAY_TASK_ID --project_name test_sweep1.1
