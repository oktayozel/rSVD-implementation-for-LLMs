#!/usr/bin/env python3
"""
Command-line script for running experiments from the experiments directory.

Usage:
    python run_experiments.py --suite quick
    python run_experiments.py --suite comparison --dataset shakespeare_char
    python run_experiments.py --list-suites
"""

import argparse
import sys
from pathlib import Path

# Find project root (parent of experiments directory)
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Add src and config to path
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'config'))

import torch
from mla_gpt.utils.experiment_runner import ExperimentRunner
import experiments_config


def main():
    parser = argparse.ArgumentParser(
        description='Run experiments on GPT models with different compression methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available suites
  python run_experiments.py --list-suites
  
  # Run quick test suite
  python run_experiments.py --suite quick
  
  # Run comparison suite on shakespeare dataset
  python run_experiments.py --suite comparison --dataset shakespeare_char
  
  # Run with CPU
  python run_experiments.py --suite quick --device cpu
        """
    )
    
    parser.add_argument(
        '--suite',
        type=str,
        help='Experiment suite to run (quick, comparison, svd_comparison, mla, rank_ablation, comprehensive)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='shakespeare_char',
        help='Dataset to use (shakespeare_char, shakespeare, openwebtext). Default: shakespeare_char'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda, cpu). Default: auto-detect'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results. Default: experiments/results'
    )
    
    parser.add_argument(
        '--list-suites',
        action='store_true',
        help='List available experiment suites and exit'
    )
    
    args = parser.parse_args()
    
    # List suites if requested
    if args.list_suites:
        experiments_config.list_suites()
        return
    
    # Validate arguments
    if not args.suite:
        parser.error('--suite is required (or use --list-suites to see options)')
    
    # Determine device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Set default output directory relative to project root
    if args.output_dir is None:
        output_dir = str(project_root / 'experiments' / 'results')
    else:
        output_dir = args.output_dir
    
    print("=" * 80)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print(f"Suite: {args.suite}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    print()
    
    # Load experiment configurations
    try:
        configs = experiments_config.get_suite(args.suite)
    except ValueError as e:
        print(f"Error: {e}")
        print("\nAvailable suites:")
        experiments_config.list_suites()
        return 1
    
    print(f"Loaded {len(configs)} experiment configurations")
    print("\nExperiments to run:")
    for i, config in enumerate(configs, 1):
        print(f"{i:2d}. {config.name:30s} - {config.description}")
    print()
    
    # Ask for confirmation
    response = input("Continue with these experiments? [Y/n] ")
    if response.lower() not in ['', 'y', 'yes']:
        print("Cancelled.")
        return 0
    
    # Initialize runner
    runner = ExperimentRunner(
        data_dir=str(project_root / 'data'),
        output_dir=output_dir,
        device=device
    )
    
    # Run experiments
    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENTS")
    print("=" * 80)
    print()
    
    try:
        results = runner.run_experiments(configs, dataset=args.dataset)
        
        print("\n" + "=" * 80)
        print("EXPERIMENTS COMPLETED")
        print("=" * 80)
        print(f"\nSuccessfully completed {len(results)} experiments")
        print(f"Results saved to: {output_dir}")
        print("\nTo visualize results, open the Jupyter notebook:")
        print(f"  jupyter notebook {project_root}/notebooks/experiments_suite.ipynb")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\nError running experiments: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
