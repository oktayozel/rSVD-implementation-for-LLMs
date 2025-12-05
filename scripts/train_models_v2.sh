#!/bin/bash
# CPU training script

# Set number of threads (match your core count)
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32

# Optimal for different CPU counts:
# 4 cores:  OMP_NUM_THREADS=4
# 8 cores:  OMP_NUM_THREADS=8
# 16 cores: OMP_NUM_THREADS=16
# 32 cores: OMP_NUM_THREADS=32
# 64 cores: OMP_NUM_THREADS=64

echo "Using $OMP_NUM_THREADS CPU threads"
echo "CPU info:"
lscpu | grep -E "CPU\(s\)|Thread|Core|Socket"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_PY="${REPO_ROOT}/src/mla_gpt/cli/train.py"
LOG_DIR="${REPO_ROOT}/logs"

mkdir -p "$LOG_DIR"

train_model() {
    local config=$1
    local name=$2
    local log_file="${LOG_DIR}/${name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "=========================================="
    echo "Training: $name"
    echo "Config: $config"
    echo "=========================================="
    
    local start_time=$(date +%s)
    
    cd "$REPO_ROOT"
    python -u "$TRAIN_PY" "$config" 2>&1 | tee "$log_file"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_min=$((duration / 60))
    local duration_sec=$((duration % 60))
    
    echo "" | tee -a "$log_file"
    echo "Training completed in ${duration_min}m ${duration_sec}s" | tee -a "$log_file"
    echo "" | tee -a "$log_file"
    
    echo "$name: ${duration_min}m ${duration_sec}s" >> "${LOG_DIR}/cpu_timing_summary.txt"
}

# Clear timing summary
echo "CPU Training - rSVD Speedup Demo" > "${LOG_DIR}/cpu_timing_summary.txt"
echo "CPU Cores: $OMP_NUM_THREADS" >> "${LOG_DIR}/cpu_timing_summary.txt"
echo "================================" >> "${LOG_DIR}/cpu_timing_summary.txt"
echo "" >> "${LOG_DIR}/cpu_timing_summary.txt"

# Train models
train_model "${REPO_ROOT}/config/testing-v2/cpu_mla_rsvd.py" "cpu_mla_rsvd"
train_model "${REPO_ROOT}/config/testing-v2/cpu_mla_svd.py" "cpu_mla_svd"
train_model "${REPO_ROOT}/config/testing-v2/cpu_base.py" "cpu_base"

echo ""
echo "=========================================="
echo "CPU Training Complete!"
echo "=========================================="
cat "${LOG_DIR}/cpu_timing_summary.txt"