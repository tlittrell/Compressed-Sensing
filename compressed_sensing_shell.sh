#!/bin/bash
#SBATCH --job-name=compressed_sensing
#SBATCH --output=compressed_sensing.txt
#SBATCH --error=compressed_sensing.txt
#SBATCH -p sched_any_quicktest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000
#SBATCH --constraint="centos7"

module load engaging/gurobi/7.5.0
module load julia/0.6.0

xvfb-run -d julia CompressedSensing.jl
