#!/usr/bin/env python3
"""
Analyze and compare training results from the three models.
"""

import os
import re
import json
from pathlib import Path

def parse_log_file(log_path):
    """Extract key metrics from training log."""
    metrics = {
        'final_train_loss': None,
        'final_val_loss': None,
        'best_val_loss': None,
        'training_time': None,
        'model_params': None
    }
    
    with open(log_path, 'r') as f:
        content = f.read()
        
        # Extract final losses
        train_losses = re.findall(r'train loss ([\d.]+)', content)
        val_losses = re.findall(r'val loss ([\d.]+)', content)
        
        if train_losses:
            metrics['final_train_loss'] = float(train_losses[-1])
        if val_losses:
            metrics['final_val_loss'] = float(val_losses[-1])
            metrics['best_val_loss'] = min(float(x) for x in val_losses)
        
        # Extract training time
        time_match = re.search(r'Training completed in (\d+)m (\d+)s', content)
        if time_match:
            mins, secs = int(time_match.group(1)), int(time_match.group(2))
            metrics['training_time'] = mins * 60 + secs
        
        # Extract parameter count
        param_match = re.search(r'number of parameters: ([\d.]+)M', content)
        if param_match:
            metrics['model_params'] = float(param_match.group(1))
    
    return metrics

def main():
    logs_dir = Path('logs')
    
    models = {
        'Base Model': 'base_model',
        'MLA + SVD': 'mla_svd_model',
        'MLA + rSVD': 'mla_rsvd_model'
    }
    
    print("\n" + "=" * 70)
    print("Training Results Comparison")
    print("=" * 70 + "\n")
    
    results = {}
    
    for name, prefix in models.items():
        # Find most recent log file
        log_files = list(logs_dir.glob(f'{prefix}_*.log'))
        if not log_files:
            print(f"⚠️  No log file found for {name}")
            continue
        
        log_file = max(log_files, key=os.path.getctime)
        metrics = parse_log_file(log_file)
        results[name] = metrics
        
        print(f"{name}:")
        print(f"  Parameters:        {metrics['model_params']:.2f}M")
        print(f"  Training Time:     {metrics['training_time'] // 60}m {metrics['training_time'] % 60}s")
        print(f"  Final Train Loss:  {metrics['final_train_loss']:.4f}")
        print(f"  Final Val Loss:    {metrics['final_val_loss']:.4f}")
        print(f"  Best Val Loss:     {metrics['best_val_loss']:.4f}")
        print()
    
    # Comparative analysis
    if len(results) == 3:
        print("=" * 70)
        print("Comparative Analysis")
        print("=" * 70 + "\n")
        
        base_time = results['Base Model']['training_time']
        base_val = results['Base Model']['best_val_loss']
        
        for name in ['MLA + SVD', 'MLA + rSVD']:
            if name in results:
                time_diff = results[name]['training_time'] - base_time
                time_pct = (time_diff / base_time) * 100
                
                val_diff = results[name]['best_val_loss'] - base_val
                val_pct = (val_diff / base_val) * 100
                
                print(f"{name} vs Base:")
                print(f"  Time difference:    {time_diff:+.0f}s ({time_pct:+.1f}%)")
                print(f"  Val loss difference: {val_diff:+.4f} ({val_pct:+.1f}%)")
                print()
        
        # Compare SVD vs rSVD
        if 'MLA + SVD' in results and 'MLA + rSVD' in results:
            svd_time = results['MLA + SVD']['training_time']
            rsvd_time = results['MLA + rSVD']['training_time']
            time_diff = rsvd_time - svd_time
            time_pct = (time_diff / svd_time) * 100
            
            svd_val = results['MLA + SVD']['best_val_loss']
            rsvd_val = results['MLA + rSVD']['best_val_loss']
            val_diff = rsvd_val - svd_val
            
            print("MLA + rSVD vs MLA + SVD:")
            print(f"  Time difference:    {time_diff:+.0f}s ({time_pct:+.1f}%)")
            print(f"  Val loss difference: {val_diff:+.4f}")
            print()
    
    # Save results as JSON
    output_file = logs_dir / 'comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print()

if __name__ == '__main__':
    main()