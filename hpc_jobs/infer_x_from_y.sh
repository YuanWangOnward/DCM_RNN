#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=2:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=infer_x
#SBATCH --mail-type=END
#SBATCH --mail-user=yw1225@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load python3/intel/3.5.3
# module load tensorflow/python3.5/1.1.0
module load tensorflow/python3.5/1.0.1
#SBATCH --gres=gpu:1

JOBNAME=infer_x_from_y_nd_0_rhp_0_rhi_o_ns_0_smooth_02
RUNDIR=$SCRATCH/runs/$JOBNAME-${SLURM_JOB_ID/.*}
SOURCEDIR=~/projects/DCM_RNN/dcm_rnn
OUTPUTDIR=$SCRATCH/results/DCM_RNN/$JOBNAME/${SLURM_JOB_ID/.*}
#SBATCH --gres=gpu:1


mkdir -p $RUNDIR
mkdir -p $OUTPUTDIR
export PYTHONPATH=$PYTHONPATH:$SOURCEDIR
echo $PYTHONPATH
cd $RUNDIR

python3 ~/projects/DCM_RNN/experiments/infer_x_from_y/experiment_main.py


# leave a blank line at the end
