#!/usr/bin/env bash
#SBATCH --mem  128G
#SBATCH --gres gpu:2
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 6
#SBATCH --constrain "khazadum"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shafner@kth.se
#SBATCH --output=/home/s/h/shafner/damage_detection/output/logs/slurm/%j.out

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. /home/s/h/shafner/anaconda3/etc/profile.d/conda.sh
# modify this to activate your environment
conda activate dmg
cd /home/s/h/shafner/repos/building_damage

srun python /home/s/h/shafner/repos/building_damage/main.py  trainer.devices=2 data.num_workers=6 group="test"