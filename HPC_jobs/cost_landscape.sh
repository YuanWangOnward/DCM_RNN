#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --time=48:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=cost_landscape
#SBATCH --mail-type=END
#SBATCH --mail-user=yw1225@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load python3/intel/3.5.3

JOBNAME=t_delta
RUNDIR=$SCRATCH/runs/$JOBNAME-${SLURM_JOB_ID/.*}
SOURCEDIR=~/projects/DCM_RNN/dcm_rnn
OUTPUTDIR=$SCRATCH/results/DCM_RNN/cost_landscape/${SLURM_JOB_ID/.*}
SOURCEDIR=~/projects/DCM_RNN/dcm_rnn

export PYTHONPATH=$PYTHONPATH:$SOURCEDIR
mkdir -p $RUNDIR
mkdir -p $OUTPUTDIR
cd $RUNDIR

# now start the job:
python3 ~/projects/DCM_RNN/experiments/cost_landscape/partial_free_parameters.py  -v -o $OUTPUTDIR -s $SOURCEDIR

# leave a blank line at the end
