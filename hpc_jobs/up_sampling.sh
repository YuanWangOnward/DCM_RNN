#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=10
#SBATCH --time=48:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=up_sampling
#SBATCH --mail-type=END
#SBATCH --mail-user=yw1225@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load python3/intel/3.5.3

JOBNAME=up_sampling
RUNDIR=$SCRATCH/runs/$JOBNAME-${SLURM_JOB_ID/.*}
DATADIR=$SCRATCH/data/DCM_RNN/simulated_data
OUTPUTDIR=$SCRATCH/results/DCM_RNN/up_sampling/run-${SLURM_JOB_ID/.*}
SOURCEDIR=~/projects/DCM_RNN/dcm_rnn

export PYTHONPATH=$PYTHONPATH:$SOURCEDIR
mkdir -p $RUNDIR
mkdir -p $OUTPUTDIR
cd $RUNDIR

# now start the job:
python3 ~/projects/DCM_RNN/experiments/up_sampling/experiment_main.py  -i $DATADIR -o $OUTPUTDIR

# leave a blank line at the end
