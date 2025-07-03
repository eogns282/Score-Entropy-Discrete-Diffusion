"""
Run focused scaled experiments to demonstrate Enhanced AEGUD improvements.
This script runs a smaller but comprehensive set of experiments.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from experiments.aegud.run_scaled_experiments import (
    create_scaled_model, create_synthetic_dataloader,
    ScaledTrainer, run_scaled_experiment
)


def run_focused_experiments(device='cuda:0'):
    """Run a focused set of experiments with moderate scale."""
    
    # Focused configuration - smaller scale for faster results
    base_config = {
        'seed': 42,
        'vocab_size': 500,  # Moderate vocabulary
        'seq_len': 64,      # Shorter sequences
        'batch_size': 128,
        'num_steps': 5000,  # Fewer steps for quick results
        'num_epochs': 100,
        'train_batches_per_epoch': 200,
        
        'model': {
            'hidden_size': 256,  # Smaller model
            'n_heads': 8,
            'n_blocks': 6,
            'cond_dim': 128,
            'dropout': 0.1,
            'scale_by_sigma': False
        },
        
        'noise_type': 'geometric',
        'noise_params': {
            'sigma_min': 0.001,
            'sigma_max': 50.0
        },
        
        'optimizer': {
            'lr': 5e-4,
            'min_lr': 1e-6,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.01,
            'max_steps': 5000
        },
        
        'log_every': 50,
        'val_every': 250,
        'checkpoint_every': 1000,
        'convergence_test_every': 500,
        
        'use_wandb': False
    }
    
    # Define focused experiment set
    experiments = {
        'Original_AEGUD': {
            **base_config,
            'graph_type': 'original_aegud',
            'entropy_scale': 1.0,
            'sparsity_k': 50,
            'enhancement_params': {}
        },
        
        'Enhanced_AEGUD_Best': {
            **base_config,
            'graph_type': 'enhanced_aegud',
            'entropy_scale': 1.0,
            'sparsity_k': 50,
            'enhancement_params': {
                'use_asymptotic_guarantee': True,
                'use_two_stage': True,
                'stage_transition_point': 0.8,
                'use_controlled_decay': True,
                'decay_tau': 0.15,  # Slightly slower decay
                'kl_regularization_weight': 0.02  # Moderate regularization
            }
        },
        
        'Baseline_Uniform': {
            **base_config,
            'graph_type': 'uniform',
            'enhancement_params': {}
        }
    }
    
    results = {}
    
    print("="*80)
    print("FOCUSED SCALED EXPERIMENTS")
    print("="*80)
    print(f"Running on device: {device}")
    print(f"Vocabulary size: {base_config['vocab_size']}")
    print(f"Training steps: {base_config['num_steps']}")
    print(f"Model parameters: ~10M")
    print()
    
    for exp_name, exp_config in experiments.items():
        try:
            print(f"\nStarting experiment: {exp_name}")
            result = run_scaled_experiment(exp_name, exp_config, device)
            results[exp_name] = result
            
            # Print immediate results
            val = result.get('final_validation', {})
            conv = val.get('convergence', {}).get('convergence', {})
            
            print(f"\n{exp_name} Results:")
            print(f"  - Best Val Loss: {result.get('best_val_loss', 'N/A'):.4f}")
            print(f"  - Convergence: {'PASSED' if conv.get('converged', False) else 'FAILED'}")
            print(f"  - Final Entropy: {conv.get('entropy_ratio', 'N/A'):.4f}")
            print(f"  - Final KL: {val.get('convergence', {}).get('final_kl', 'N/A'):.6f}")
            
        except Exception as e:
            print(f"Error in experiment {exp_name}: {str(e)}")
            results[exp_name] = {'error': str(e)}
    
    # Create comparison visualization
    create_focused_comparison_plots(results)
    
    return results


def create_focused_comparison_plots(results):
    """Create focused comparison visualizations."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"experiments/aegud/scaled_results/focused_{timestamp}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics for plotting
    experiments = list(results.keys())
    
    # Create figure with comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Training Loss Curves
    ax = axes[0, 0]
    for exp_name, result in results.items():
        if 'error' not in result and 'metrics' in result:
            metrics = result['metrics']
            if 'train_loss' in metrics and 'steps' in metrics:
                ax.plot(metrics['steps'], metrics['train_loss'], 
                       label=exp_name.replace('_', ' '), linewidth=2)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 2. Convergence Metrics
    ax = axes[0, 1]
    
    final_entropies = []
    final_kls = []
    names = []
    
    for exp_name, result in results.items():
        if 'error' not in result:
            val = result.get('final_validation', {})
            conv = val.get('convergence', {})
            
            if 'metrics' in conv:
                final_entropies.append(conv['metrics']['entropy'][-1])
                final_kls.append(conv['metrics']['kl_from_uniform'][-1])
                names.append(exp_name.replace('_', ' '))
    
    if names:
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, final_entropies, width, label='Final Entropy', alpha=0.8)
        bars2 = ax.bar(x + width/2, final_kls, width, label='Final KL', alpha=0.8)
        
        ax.axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='Entropy Target')
        ax.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='KL Target')
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Value')
        ax.set_title('Final Convergence Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Convergence Evolution
    ax = axes[1, 0]
    
    for exp_name, result in results.items():
        if 'error' not in result and 'metrics' in result:
            conv_tests = result['metrics'].get('convergence_tests', [])
            if conv_tests:
                steps = [t['step'] for t in conv_tests]
                kls = [t['final_kl'] for t in conv_tests]
                ax.plot(steps, kls, marker='o', label=exp_name.replace('_', ' '), linewidth=2)
    
    ax.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='Target')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('KL Divergence from Uniform')
    ax.set_title('Convergence Evolution During Training')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Summary Table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary data
    summary_data = [['Metric', 'Original\nAEGUD', 'Enhanced\nAEGUD', 'Baseline\nUniform']]
    
    # Extract metrics
    metrics_to_show = ['Best Loss', 'Final Entropy', 'Final KL', 'Converged']
    
    for metric in metrics_to_show:
        row = [metric]
        
        for exp in ['Original_AEGUD', 'Enhanced_AEGUD_Best', 'Baseline_Uniform']:
            if exp in results and 'error' not in results[exp]:
                result = results[exp]
                val = result.get('final_validation', {})
                conv = val.get('convergence', {}).get('convergence', {})
                
                if metric == 'Best Loss':
                    value = f"{result.get('best_val_loss', 'N/A'):.3f}"
                elif metric == 'Final Entropy':
                    value = f"{conv.get('entropy_ratio', 'N/A'):.3f}"
                elif metric == 'Final KL':
                    value = f"{val.get('convergence', {}).get('final_kl', 'N/A'):.4f}"
                elif metric == 'Converged':
                    value = '✓' if conv.get('converged', False) else '✗'
                
                row.append(value)
            else:
                row.append('N/A')
        
        summary_data.append(row)
    
    table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0],
                    cellLoc='center', loc='center',
                    colWidths=[0.25, 0.25, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight convergence results
    for i in range(1, len(summary_data)):
        if summary_data[i][0] == 'Converged':
            for j in range(1, len(summary_data[i])):
                if summary_data[i][j] == '✓':
                    table[(i-1, j)].set_facecolor('#90EE90')
                else:
                    table[(i-1, j)].set_facecolor('#FFB6C1')
    
    ax.set_title('Performance Summary', fontsize=12, pad=20)
    
    plt.suptitle('Enhanced AEGUD: Scaled Experiment Results', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    plot_path = save_dir / 'comparison_plots.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plots to {plot_path}")
    
    plt.close()
    
    # Save results
    results_path = save_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run focused scaled experiments')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for experiments')
    
    args = parser.parse_args()
    
    # Run focused experiments
    results = run_focused_experiments(device=args.device)
    
    print("\n" + "="*80)
    print("FOCUSED EXPERIMENTS COMPLETED")
    print("="*80)
    
    # Print final summary
    for exp_name, result in results.items():
        if 'error' in result:
            print(f"\n{exp_name}: ERROR - {result['error']}")
        else:
            print(f"\n{exp_name}: SUCCESS")
            print(f"  Training time: {result.get('training_time', 0)/60:.2f} minutes")
    
    print("\nAll experiments completed!")