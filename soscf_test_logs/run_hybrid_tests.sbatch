#!/bin/bash
#SBATCH -J cider_hybrid_tests
#SBATCH -p test
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /n/holystore01/LABS/kozinsky_lab/Lab/User/mabdallah/CiderPress_SOSCF/soscf_test_logs/hybrid_tests_%j.out
#SBATCH -e /n/holystore01/LABS/kozinsky_lab/Lab/User/mabdallah/CiderPress_SOSCF/soscf_test_logs/hybrid_tests_%j.err

set -x
source /n/home10/mabdallah/activate_soscf_env.sh
export OMP_NUM_THREADS=16

cd /n/holystore01/LABS/kozinsky_lab/Lab/User/mabdallah/CiderPress_SOSCF

echo "=== ciderpress resolves to: ==="
python -c "import ciderpress; print(ciderpress.__file__)"
git log --oneline -1

echo "=== FAST local-hybrid suite ==="
python -m pytest ciderpress/pyscf/tests/test_local_hybrid.py -v --durations=15
FAST_STATUS=$?

echo "=== SLOW local-hybrid suite (trained models) ==="
CIDER_SLOW_TESTS=1 python -m pytest ciderpress/pyscf/tests/test_local_hybrid_slow.py -v --durations=15
SLOW_STATUS=$?

echo "=== Regression: newton + feature suites ==="
python -m pytest ciderpress/pyscf/tests/test_newton.py ciderpress/pyscf/tests/test_sl.py -v --durations=5
REG_STATUS=$?

echo "=== SUMMARY ==="
echo "fast: $FAST_STATUS  slow: $SLOW_STATUS  regression: $REG_STATUS"
