#!/bin/bash

# Model comparison script for H100 GPU
# Compares: Base model, MLA+SVD, MLA+rSVD

set -e  # Exit on error

echo "=========================================="
echo "Model Comparison Experiment"
echo "=========================================="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
echo "Date: $(date)"
echo ""

# Resolve script directory and train.py path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_PY="${REPO_ROOT}/src/mla_gpt/cli/train.py"

# Create logs directory in home
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "$LOG_DIR"

if [[ ! -f "$TRAIN_PY" ]]; then
  echo "Error: train.py not found at: $TRAIN_PY"
  echo "Please verify the repo layout."
  exit 1
fi

# Function to train and time a model
train_model() {
    local config=$1
    local name=$2
    local log_file="${LOG_DIR}/${name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "=========================================="
    echo "Training: $name"
    echo "Config: $config"
    echo "Log: $log_file"
    echo "=========================================="
    
    # Record start time
    local start_time=$(date +%s)
    
    # Change to repo root before running (train.py expects to be run from repo root)
    cd "$REPO_ROOT"
    
    # Train model and capture output
    python "$TRAIN_PY" "$config" 2>&1 | tee "$log_file"
    
    # Record end time
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_min=$((duration / 60))
    local duration_sec=$((duration % 60))
    
    echo "" | tee -a "$log_file"
    echo "Training completed in ${duration_min}m ${duration_sec}s" | tee -a "$log_file"
    echo "" | tee -a "$log_file"
    
    # Append timing summary to overall log
    echo "$name: ${duration_min}m ${duration_sec}s" >> "${LOG_DIR}/timing_summary.txt"
}

# Clear previous timing summary
echo "Training Time Summary" > "${LOG_DIR}/timing_summary.txt"
echo "====================" >> "${LOG_DIR}/timing_summary.txt"
echo "" >> "${LOG_DIR}/timing_summary.txt"

# Train models sequentially (configs resolved relative to repo root)
train_model "${REPO_ROOT}/config/testing/base_nanogpt.py" "base_nanogpt"
train_model "${REPO_ROOT}/config/testing/mla_qkv_svd.py" "mla_qkv_svd"
train_model "${REPO_ROOT}/config/testing/mla_qkv_rsvd.py" "mla_qkv_rsvd"

# Final summary
echo ""
echo "=========================================="
echo "All training completed!"
echo "=========================================="
echo ""
cat "${LOG_DIR}/timing_summary.txt"
echo ""
echo "Logs saved in: ${LOG_DIR}/"
echo "Model checkpoints:"
echo "  - Base:      out-shakespeare-char-base/"
echo "  - MLA+SVD:   out-shakespeare-char-mla-qkv-svd/"
echo "  - MLA+rSVD:  out-shakespeare-char-mla-qkv-rsvd/"
echo ""