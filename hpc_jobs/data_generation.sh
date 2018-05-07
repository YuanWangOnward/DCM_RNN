#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --time=48:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=data_generation
#SBATCH --mail-type=END
#SBATCH --mail-user=yw1225@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load python3/intel/3.5.3

RUNDIR=$SCRATCH/runs/run-${SLURM_JOB_ID/.*}
SOURCEDIR=~/projects/DCM_RNN/dcm_rnn
OURPUTDIR=$RUNDIR/simulated_data

export PYTHONPATH=$PYTHONPATH:$SOURCEDIR
mkdir -p $RUNDIR
mkdir -p $OURPUTDIR
cd $RUNDIR

python3 ~/projects/DCM_RNN/dcm_rnn/data_generation.py -o $OURPUTDIR


# leave a blank line at the end
