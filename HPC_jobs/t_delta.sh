#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=120:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=t_delta
#SBATCH --mail-type=END
#SBATCH --mail-user=yw1225@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load python3/intel/3.5.3

JOBNAME=t_delta
RUNDIR=$SCRATCH/runs/$JOBNAME-${SLURM_JOB_ID/.*}
DATADIR=$SCRATCH/data/DCM_RNN/generated
OUTPUTDIR=$SCRATCH/results/DCM_RNN/t_delta/run-${SLURM_JOB_ID/.*}

mkdir -p $RUNDIR
mkdir -p $OUTPUTDIR
cd $RUNDIR

# now start the job:
# copy needed source files, because of import problem (pool.map) multiple copies are need
mkdir -p $RUNDIR/dcm_rnn
cp ~/projects/DCM_RNN/dcm_rnn/toolboxes.py $RUNDIR
cp ~/projects/DCM_RNN/dcm_rnn/toolboxes.py $RUNDIR/dcm_rnn/toolboxes.py
cp ~/projects/DCM_RNN/dcm_rnn/database_toolboxes.py $RUNDIR
cp ~/projects/DCM_RNN/dcm_rnn/database_toolboxes.py $RUNDIR/dcm_rnn/database_toolboxes.py

cd $RUNDIR
python3 ~/projects/DCM_RNN/experiments/t_delta/experiment_main.py  -i $DATADIR -o $OUTPUTDIR

# remove source files
rm $RUNDIR/toolboxes.py
rm $RUNDIR/database_toolboxes.py
rm $RUNDIR/dcm_rnn/toolboxes.py
rm $RUNDIR/dcm_rnn/database_toolboxes.py


# leave a blank line at the end
