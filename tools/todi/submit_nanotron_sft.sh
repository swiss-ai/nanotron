#!/bin/bash

#SBATCH --job-name nanotron_sft
#SBATCH --chdir /users/asolergi/SFT/nanotron # TODO Set this path!!!
#SBATCH --output reports/R-%x.%j.out    # Make sure this paths exists, otherwise the job will fail silently
#SBATCH --error reports/R-%x.%j.err     # Make sure this paths exists, otherwise the job will fail silently
#SBATCH --nodes 4                       # number of Nodes
#SBATCH --ntasks-per-node 1             # number of MP tasks. IMPORTANT: torchrun represents just 1 Slurm task
#SBATCH --gres gpu:4                    # Number of GPUs
#SBATCH --cpus-per-task 288             # number of CPUs per task.
#SBATCH --time 11:59:59                 # maximum execution time (DD-HH:MM:SS). Mandatory field in MN5
#SBATCH --reservation todi
#SBATCH --environment  /store/swissai/a06/.sft_toni/nanotron_sft.toml
#SBATCH --contiguous

echo "START TIME: $(date)"

# auto-fail on any errors in this script
set -eo pipefail

# logging script's variables/commands for future debug needs
set -x

######################
### Set environment ###
######################
GPUS_PER_NODE=4
echo "NODES: $SLURM_NNODES"
######################

######################
#### Set network #####
######################
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
######################

# note that we don't want to interpolate `\$SLURM_PROCID` till `srun` since otherwise all nodes will get
# 0 and the launcher will hang
#
# same goes for `\$(hostname -s|tr -dc '0-9')` - we want it to interpolate at `srun` time
LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    --node_rank ${SLURM_PROCID} \
    "

PYTHON_FILE=/workspace/nanotron/run_train.py
NANOTRON_CONFIG=/users/asolergi/SFT/nanotron/examples/config_llama8b_sft.yaml # TODO Set this path!!!

export CMD="CUDA_DEVICE_MAX_CONNECTIONS=1 $LAUNCHER $PYTHON_FILE --config $NANOTRON_CONFIG"

echo $CMD

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
SRUN_ARGS=" \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --jobid $SLURM_JOB_ID \
    --wait 60 \
    --unbuffered \
    "

# bash -c is needed for the delayed interpolation of env vars to work
srun $SRUN_ARGS bash -c "$CMD"

echo "END TIME: $(date)"
