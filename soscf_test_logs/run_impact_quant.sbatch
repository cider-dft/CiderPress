#!/bin/bash
#SBATCH -J nldf_impact
#SBATCH -p test
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /n/holystore01/LABS/kozinsky_lab/Lab/User/mabdallah/CiderPress_SOSCF/soscf_test_logs/nldf_impact_%j.out
#SBATCH -e /n/holystore01/LABS/kozinsky_lab/Lab/User/mabdallah/CiderPress_SOSCF/soscf_test_logs/nldf_impact_%j.err
source /n/home10/mabdallah/activate_soscf_env.sh
export OMP_NUM_THREADS=16
cd /n/holystore01/LABS/kozinsky_lab/Lab/User/mabdallah/CiderPress_SOSCF
python soscf_test_logs/quantify_nldf_uks_impact.py
