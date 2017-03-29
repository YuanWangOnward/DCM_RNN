#!/bin/bash
#PBS -l nodes=1:ppn=2
#PBS -l walltime=01:00:00
#PBS -l mem=1GB
#PBS -N remove_anaconda
#PBS -M yw1225@nyu.edu
#PBS -j oe

module purge

module load python3/intel/3.5.1
module load matplotlib/intel/1.5.3
module load tensorflow/python3.5.1/20161029

# module load python3/intel/3.5.1

module load matplotlib/intel/1.5.3
module load tensorflow/python2.7/20161207



## module load numpy/intel/1.12.0

# python3 ~/projects/DCM-RNN/HPC_jobs/import_test.py

# job_name=remove_anaconda

# RUNDIR=$SCRATCH/DCM_RNN/$job_name-${PBS_JOBID/.*}
# mkdir -p $RUNDIR

# DATADIR=$SCRATCH/DCM_RNN/data
# cd $RUNDIR

# cd $RUNDIR
# touch output.txt
# echo "HPC job runs sucessfully" > $RUNDIR/output.txt

# leave a blank line at the end