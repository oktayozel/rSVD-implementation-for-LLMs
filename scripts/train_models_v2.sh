#!/bin/bash

# Model comparison script for WikiText-103 dataset
# Compares: Base (350M), MLA+SVD (350M), MLA+rSVD (350M)
# Goal: Demonstrate rSVD speed advantage

set -e

echo "=========================================="
echo "WikiText-103 Model Comparison"
echo "=========================================="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
echo "Date: $(date)"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_PY="${REPO_ROOT}/src/mla_gpt/cli/train.py"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "$LOG_DIR"

if [[ ! -f "$TRAIN_PY" ]]; then
  echo "Error: train.py not found at: $TRAIN_PY"
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
    
    echo "$name: ${duration_min}m ${duration_sec}s" >> "${LOG_DIR}/timing_summary_v2.txt"
}

# Clear previous timing summary
echo "WikiText-103 Training Time Summary" > "${LOG_DIR}/timing_summary_v2.txt"
echo "===================================" >> "${LOG_DIR}/timing_summary_v2.txt"
echo "" >> "${LOG_DIR}/timing_summary_v2.txt"

# Train models sequentially
train_model "${REPO_ROOT}/config/testing-v2/base_nanogpt.py" "base_350M"
train_model "${REPO_ROOT}/config/testing-v2/mla_svd.py" "mla_svd_350M"
train_model "${REPO_ROOT}/config/testing-v2/mla_rsvd.py" "mla_rsvd_350M"

# Final summary
echo ""
echo "=========================================="
echo "All training completed!"
echo "=========================================="
echo ""
cat "${LOG_DIR}/timing_summary_v2.txt"
echo ""
echo "Logs saved in: ${LOG_DIR}/"
echo "Model checkpoints:"
echo "  - Base:      ${REPO_ROOT}/out-wikitext103-base/"
echo "  - MLA+SVD:   ${REPO_ROOT}/out-wikitext103-mla-svd/"
echo "  - MLA+rSVD:  ${REPO_ROOT}/out-wikitext103-mla-rsvd/"
echo ""
echo "Expected Results:"
echo "  - Base: Slowest training, baseline perplexity"
echo "  - MLA+SVD: Faster than base (smaller KV cache), but SVD adds overhead"
echo "  - MLA+rSVD: FASTEST training, similar perplexity to MLA+SVD"
echo ""