#! /bin/bash

#Batch Job Paremeters
#SBATCH --partition=gp2d
#SBATCH --account=GOV108017
##SBATCH -o job.%j.out # Name of stdout output file (%j expands to jobId)
#SBATCH --gres=gpu:2
#SBATCH --nodes=3
#SBATCH --mem=8192
##SBATCH --nodelist=gn1201.twcc.ai,gn1204.twcc.ai,gn1205.twcc.ai
#SBATCH --cpus-per-task=2

#Operations
echo “Hello World”
date
hostname
nvidia-smi
#sleep 5
echo “Bye”
