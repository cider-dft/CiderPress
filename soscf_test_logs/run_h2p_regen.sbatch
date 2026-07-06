#!/bin/bash
#SBATCH -J h2p_regen
#SBATCH -p test
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o /n/holystore01/LABS/kozinsky_lab/Lab/User/mabdallah/CiderPress_SOSCF/soscf_test_logs/h2p_regen_%j.out
#SBATCH -e /n/holystore01/LABS/kozinsky_lab/Lab/User/mabdallah/CiderPress_SOSCF/soscf_test_logs/h2p_regen_%j.err
source /n/home10/mabdallah/activate_soscf_env.sh
export OMP_NUM_THREADS=16
export CIDER_SLOW_TESTS=1
cd /n/holystore01/LABS/kozinsky_lab/Lab/User/mabdallah/CiderPress_SOSCF
python soscf_test_logs/regen_h2p_reference.py
