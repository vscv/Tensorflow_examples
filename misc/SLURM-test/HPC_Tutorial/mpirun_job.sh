#!/bin/bash
#SBATCH -J horovod_test            # Job name
#SBATCH -o %j.horovod.out          # Name of stdout output file (%j expands to jobId)
#SBATCH -t 00:10:00                # Run time (hh:mm:ss)
#SBATCH --account=GOV108017        #iService Project id
#SBATCH --nodes=2                  # Number of nodes
#SBATCH --ntasks-per-node=8        # Number of MPI process per node
#SBATCH --gres=gpu:8               # Number of GPUs per node

module purge
module load compiler/gnu/7.3.0 openmpi3 singularity
cmd="singularity exec --nv tensorflow_19.02-py3.sif python tensorflow_synthetic_benchmark.py --batch-size 256"
mpirun $cmd
