#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=48:00:00
#SBATCH --mem=6GB
#SBATCH --job-name=data_generation
#SBATCH --mail-type=END
#SBATCH --mail-user=yw1225@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load python3/intel/3.5.3

RUNDIR=$SCRATCH/data_generation/run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR
cd $RUNDIR

# now start the Stata job:
mkdir -p $RUNDIR/DCM_RNN
cp -a ~/projects/DCM_RNN $RUNDIR
cd $RUNDIR/DCM_RNN
python3 dcm_rnn/data_generation.py


# leave a blank line at the end
