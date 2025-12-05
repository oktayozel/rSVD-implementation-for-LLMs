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

# Create logs directory
mkdir -p logs

# Prepare dataset
echo "Preparing Shakespeare dataset..."
python data/shakespeare_char/prepare.py
echo ""

# Function to train and time a model
train_model() {
    local config=$1
    local name=$2
    local log_file="logs/${name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "=========================================="
    echo "Training: $name"
    echo "Config: $config"
    echo "Log: $log_file"
    echo "=========================================="
    
    # Record start time
    local start_time=$(date +%s)
    
    # Train model and capture output
    python -m mla_gpt.cli.train "$config" 2>&1 | tee "$log_file"
    
    # Record end time
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_min=$((duration / 60))
    local duration_sec=$((duration % 60))
    
    echo "" | tee -a "$log_file"
    echo "Training completed in ${duration_min}m ${duration_sec}s" | tee -a "$log_file"
    echo "" | tee -a "$log_file"
    
    # Append timing summary to overall log
    echo "$name: ${duration_min}m ${duration_sec}s" >> logs/timing_summary.txt
}

# Clear previous timing summary
echo "Training Time Summary" > logs/timing_summary.txt
echo "====================" >> logs/timing_summary.txt
echo "" >> logs/timing_summary.txt

# Train models sequentially
train_model "config/testing/base_nanogpt.py" "base_nanogpt"
train_model "config/testing/mla_qkv_svd.py" "mla_qkv_svd"
train_model "config/testing/mla_qkv_rsvd.py" "mla_qkv_rsvd"

# Final summary
echo ""
echo "=========================================="
echo "All training completed!"
echo "=========================================="
echo ""
cat logs/timing_summary.txt
echo ""
echo "Logs saved in: logs/"
echo "Model checkpoints:"
echo "  - Base:      out-shakespeare-char-base/"
echo "  - MLA+SVD:   out-shakespeare-char-mla-qkv-svd/"
echo "  - MLA+rSVD:  out-shakespeare-char-mla-qkv-rsvd/"
echo ""