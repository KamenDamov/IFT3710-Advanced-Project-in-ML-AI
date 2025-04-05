#!/bin/bash
#SBATCH --job-name=llm_finetune  # This changes the job's display name when you run the "squeue" command
#SBATCH --account=def-[sponsor_name]    # Allocation identifier
#SBATCH --time=0-24:00:00        # Adjust this to match the walltime of your job and according to the compute node partitions, more details below
#SBATCH --nodes=1                # Use only one compute node (here we use only one server)
#SBATCH --ntasks=1               # For running several script in parallel in one script, change this. Refer to https://stackoverflow.com/questions/39186698/what-does-the-ntasks-or-n-tasks-does-in-slurm
#SBATCH --gpus-per-node=a100:1   # The number and type of GPU to use in each compute node.
#SBATCH --cpus-per-gpu=12        # Please set this according to the Core Equivalent Bundles, more details below
#SBATCH --mem-per-gpu=124.5G     # Please set this according to the Core Equivalent Bundles, more details below
#SBATCH --mail-type=ALL          # Receive emails when job begins, ends or fails
#SBATCH --mail-user=your_mail@umontreal.ca  # Where to receive all the job-related emails

# Set the path to data and model dir
PROJECT_DIR=/home/kamendam/projects/def-bangliu/kamendam
DATA_DIR=$PROJECT_DIR/data/
MODEL_DIR=$PROJECT_DIR/models/llama
REPO_DIR=$SCRATCH/llama_finetune_code

# Load necessary modules
module load python/3.11.9 cuda/11.7 scipy-stack gcc/9.3.0 arrow/8

# Generate your virtual environment in $SLURM_TMPDIR, don't change this line
virtualenv --no-download "${SLURM_TMPDIR}"/my_env && source "${SLURM_TMPDIR}"/my_env/bin/activate

# Install packages on the virtualenv, please always have the no-index argument
pip install --no-index -r "${REPO_DIR}"/requirements.txt

# Prepare data to the compute node's local storage. Saving tarballs in /project saves space, reduces file counts and accelerates this data transfer process
mkdir $SLURM_TMPDIR/data
tar -xzvf $DATA_DIR/data.tar.gz -C $SLURM_TMPDIR/data

# Setup wandb (requires WANDB_API_KEY setup in your ~/.bash_profile)
wandb login "$WANDB_API_KEY"
wandb offline

# Prepare for your experiment
cd "$REPO_DIR"

# Run your commands
accelerate launch cgan.py #--model $MODEL_DIR --dataset $SLURM_TMPDIR/data