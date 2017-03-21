#!/usr/bin/env bash

#PBS -l nodes=1:ppn=2
#PBS -l walltime=00:01:00
#PBS -l mem=8GB
#PBS -N infer_x_from_y_
#PBS -M yw1225@nyu.edu
#PBS -j oe

# first we ensure a clean running environment:
module purge
# and load the module for the software we are using:
module load tensorflow/python3.5.1/20161029
module load numpy/intel/1.12.0
module load matplotlib/intel/1.5.3

# module load os
# module load importlib
# module load copy
# module load pickle
# module load datetime

RUNDIR=$SCRATCH/DCM_RNN/run-${PBS_JOBID/.*}
DATADIR=$SCRATCH/DCM_RNN/data
mkdir -p $RUNDIR
cd $RUNDIR

# do some jobs
touch This_shell_script_runs.txt

# leave a blank line at the end






