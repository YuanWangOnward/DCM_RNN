#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=00:10:00
#SBATCH --mem=1GB
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