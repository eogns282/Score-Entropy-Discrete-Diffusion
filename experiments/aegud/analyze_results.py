#!/usr/bin/env python3
"""
Analyze AEGUD experiment results
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_results(results_dir):
    """Load all experiment results from directory."""
    results_dir = Path(results_dir)
    all_results = {}
    
    for result_file in results_dir.glob("*.json"):
        if "comparison" not in result_file.name:
            with open(result_file, 'r') as f:
                data = json.load(f)
                exp_name = data.get('exp_name', result_file.stem.split('_')[0])
                all_results[exp_name] = data
                
    return all_results

def create_comparison_plot(all_results, output_dir):
    """Create comprehensive comparison plots."""
    output_dir = Path(output_dir)
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Training Loss Curves
    ax = axes[0, 0]
    for exp_name, results in all_results.items():
        if 'losses' in results and results['losses']:
            steps = np.arange(0, len(results['losses']) * 50, 50)
            ax.plot(steps, results['losses'], label=exp_name, linewidth=2)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Final Loss Comparison
    ax = axes[0, 1]
    exp_names = []
    final_losses = []
    colors = ['blue', 'green', 'red', 'purple']
    
    for exp_name, results in all_results.items():
        if 'final_loss' in results and results['final_loss'] is not None:
            exp_names.append(exp_name.replace('_', '\n'))
            final_losses.append(results['final_loss'])
            
    bars = ax.bar(exp_names, final_losses, color=colors[:len(exp_names)])
    ax.set_ylabel('Final Loss')
    ax.set_title('Final Loss Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, loss in zip(bars, final_losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{loss:.1f}', ha='center', va='bottom')
    
    # 3. Evaluation Loss Curves
    ax = axes[1, 0]
    for exp_name, results in all_results.items():
        if 'eval_losses' in results and results['eval_losses']:
            eval_steps = np.arange(200, len(results['eval_losses']) * 200 + 200, 200)
            ax.plot(eval_steps[:len(results['eval_losses'])], 
                   results['eval_losses'], 
                   label=exp_name, 
                   linewidth=2, 
                   marker='o')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Evaluation Loss')
    ax.set_title('Evaluation Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Improvement Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate improvements
    baseline_uniform = next((r['final_loss'] for n, r in all_results.items() 
                           if 'baseline_uniform' in n), None)
    baseline_absorb = next((r['final_loss'] for n, r in all_results.items() 
                          if 'baseline_absorb' in n), None)
    adaptive_uniform = next((r['final_loss'] for n, r in all_results.items() 
                           if 'adaptive_uniform' in n), None)
    
    summary_text = "AEGUD Results Summary\n" + "="*30 + "\n\n"
    
    if baseline_uniform:
        summary_text += f"Baseline Uniform: {baseline_uniform:.1f}\n"
    if baseline_absorb:
        summary_text += f"Baseline Absorb: {baseline_absorb:.1f}\n"
    if adaptive_uniform:
        summary_text += f"Adaptive Uniform: {adaptive_uniform:.1f}\n\n"
        
        if baseline_uniform and adaptive_uniform < baseline_uniform:
            improvement = (baseline_uniform - adaptive_uniform) / baseline_uniform * 100
            summary_text += f"Improvement over Uniform: {improvement:.1f}%\n"
            
        if baseline_absorb and adaptive_uniform < baseline_absorb:
            improvement = (baseline_absorb - adaptive_uniform) / baseline_absorb * 100
            summary_text += f"Improvement over Absorb: {improvement:.1f}%\n"
            
        if adaptive_uniform < min(baseline_uniform or float('inf'), 
                                 baseline_absorb or float('inf')):
            summary_text += "\n🎉 ADAPTIVE UNIFORM WINS! 🎉"
    
    ax.text(0.1, 0.5, summary_text, fontsize=14, 
            verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = output_dir / f"aegud_analysis_{timestamp}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis plot saved to: {plot_file}")
    
    return plot_file

def print_detailed_analysis(all_results):
    """Print detailed analysis of results."""
    print("\n" + "="*80)
    print("DETAILED AEGUD EXPERIMENT ANALYSIS")
    print("="*80)
    
    # Sort experiments by final loss
    sorted_exps = sorted(
        [(name, res.get('final_loss', float('inf'))) 
         for name, res in all_results.items()],
        key=lambda x: x[1]
    )
    
    print("\nRanking by Final Loss:")
    print("-"*50)
    for i, (name, loss) in enumerate(sorted_exps, 1):
        if loss != float('inf'):
            print(f"{i}. {name:<25} Loss: {loss:.2f}")
    
    # Calculate statistics
    print("\n" + "-"*50)
    print("Statistical Summary:")
    print("-"*50)
    
    for exp_name, results in all_results.items():
        if 'losses' in results and results['losses']:
            losses = results['losses']
            print(f"\n{exp_name}:")
            print(f"  Mean Loss: {np.mean(losses):.2f}")
            print(f"  Std Loss: {np.std(losses):.2f}")
            print(f"  Min Loss: {np.min(losses):.2f}")
            print(f"  Max Loss: {np.max(losses):.2f}")
            print(f"  Final Loss: {results.get('final_loss', 'N/A')}")
            print(f"  Training Time: {results.get('total_time', 'N/A'):.1f}s")
    
    # Best result
    if sorted_exps and sorted_exps[0][1] != float('inf'):
        print("\n" + "="*50)
        print(f"WINNER: {sorted_exps[0][0]} with loss {sorted_exps[0][1]:.2f}")
        print("="*50)

def main():
    """Main analysis function."""
    results_dir = Path("experiments/aegud/results")
    
    # Load all results
    all_results = load_results(results_dir)
    
    if not all_results:
        print("No results found!")
        return
        
    print(f"Found {len(all_results)} experiments")
    
    # Create plots
    create_comparison_plot(all_results, results_dir)
    
    # Print analysis
    print_detailed_analysis(all_results)
    
    # Save summary
    summary_file = results_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(summary_file, 'w') as f:
        f.write("AEGUD EXPERIMENT SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        for exp_name, results in all_results.items():
            f.write(f"{exp_name}:\n")
            f.write(f"  Final Loss: {results.get('final_loss', 'N/A')}\n")
            f.write(f"  Training Time: {results.get('total_time', 'N/A')}\n")
            f.write(f"  Graph Type: {results.get('graph_type', 'N/A')}\n")
            f.write(f"  Noise Type: {results.get('noise_type', 'N/A')}\n\n")
    
    print(f"\nSummary saved to: {summary_file}")

if __name__ == "__main__":
    main()