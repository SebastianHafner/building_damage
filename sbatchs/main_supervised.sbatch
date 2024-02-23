#!/usr/bin/env bash
#SBATCH --output /home/s/h/shafner/damage_detection/output/logs/%J_slurm.out
#SBATCH --error  /home/s/h/shafner/damage_detection/output/logs/%J_slurm.err
#SBATCH --mem-per-cpu 4600M
#SBATCH --gres gpu:2
#SBATCH --cpus-per-task 16

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. /home/s/h/shafner/anaconda3/etc/profile.d/conda.sh

conda activate dmg
cd /home/s/h/shafner/repos/building_damage

srun python /home/s/h/shafner/repos/building_damage/main.py  trainer.devices=2 data.num_workers=8 group="test"