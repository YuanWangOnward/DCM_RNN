#!/bin/bash
#PBS -l nodes=1:ppn=2
#PBS -l walltime=00:15:00
#PBS -l mem=1GB
#PBS -N test
#PBS -M yw1225@nyu.edu
#PBS -j oe

module purge

module load python3/intel/3.5.3

job_name=data_creation

python3

# RUNDIR=$SCRATCH/DCM_RNN/$job_name-${PBS_JOBID/.*}
# mkdir -p $RUNDIR

# DATADIR=$SCRATCH/DCM_RNN/data
# cd $RUNDIR

# cd $RUNDIR
# touch output.txt
# echo "HPC job runs sucessfully" > $RUNDIR/output.txt

# leave a blank line at the end