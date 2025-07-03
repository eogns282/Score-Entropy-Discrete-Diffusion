"""
Run experiments comparing original AEGUD with enhanced versions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import argparse
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import SEDD components
from model.transformer import SEDD
from noise_lib import GeometricNoise, LogLinearNoise
from graph_lib import Uniform, Absorbing

# Import AEGUD components
from experiments.aegud.src.adaptive_uniform_graph import AdaptiveUniform
from experiments.aegud.src.enhanced_adaptive_uniform_graph import EnhancedAdaptiveUniform, create_enhanced_aegud
from experiments.aegud.src.entropy_estimator import EntropyEstimator, AdaptiveTransitionMatrix
from experiments.aegud.src.simple_loss import SimpleLoss
from experiments.aegud.src.diffusion_validator import DiffusionValidator, run_comprehensive_validation


class SimpleTrainer:
    """Simple trainer for proof-of-concept experiments."""
    
    def __init__(self, model, graph, noise, loss_fn, device='cuda'):
        self.model = model.to(device)
        self.graph = graph
        self.noise = noise
        self.loss_fn = loss_fn
        self.device = device
        self.optimizer = torch.optim.Adam(self.get_all_params(), lr=1e-3)
        
    def get_all_params(self):
        """Get all trainable parameters including graph parameters."""
        params = list(self.model.parameters())
        if hasattr(self.graph, 'parameters'):
            params.extend(self.graph.parameters())
        return params
    
    def train_step(self, x_0, t=None):
        """Single training step."""
        self.optimizer.zero_grad()
        
        # Random time if not specified
        if t is None:
            t = torch.rand(1, device=self.device).item()
        
        # Get noise level for batch
        batch_size = x_0.shape[0]
        t_batch = torch.full((batch_size,), t, device=self.device)
        sigma, dsigma = self.noise(t_batch)
        
        # Forward diffusion
        x_t = self.graph.sample_transition(x_0, sigma[:, None])
        
        # Model forward pass
        score = self.model(x_t, sigma)
        
        # Compute loss
        if hasattr(self.graph, 'score_entropy_with_kl'):
            # Enhanced graph with KL regularization
            loss = self.loss_fn(score, x_t, x_0, sigma, self.graph, t=t)
        else:
            loss = self.loss_fn(score, x_t, x_0, sigma, self.graph)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


def create_simple_model(vocab_size, hidden_dim=128, num_layers=4, device='cuda'):
    """Create a simple transformer model for testing."""
    # Create config for SEDD model
    config = {
        'model': {
            'hidden_size': hidden_dim,
            'n_heads': 4,
            'n_blocks': num_layers,
            'cond_dim': 128,
            'dropout': 0.0,
            'scale_by_sigma': False
        },
        'tokens': vocab_size,
        'graph': {
            'type': 'uniform'  # Will be overridden by the graph we pass
        }
    }
    
    model = SEDD(config)
    return model


def run_experiment(experiment_name, graph, model, noise, num_steps=1000, 
                  vocab_size=100, batch_size=32, seq_len=64, device='cuda'):
    """Run a single experiment configuration."""
    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name}")
    print(f"{'='*60}")
    
    # Create loss function
    if hasattr(graph, 'score_entropy_with_kl'):
        # Use enhanced loss with KL regularization
        def loss_fn(score, x_t, x_0, sigma, graph, t=None):
            return graph.score_entropy_with_kl(score, sigma, x_t, x_0, t=t, max_t=1.0).mean()
    else:
        loss_fn = SimpleLoss()
    
    # Create trainer
    trainer = SimpleTrainer(model, graph, noise, loss_fn, device)
    
    # Training metrics
    losses = []
    convergence_metrics = []
    
    # Validation points
    validation_steps = [0, num_steps//4, num_steps//2, 3*num_steps//4, num_steps-1]
    
    # Training loop
    start_time = time.time()
    
    for step in range(num_steps):
        # Generate batch
        x_0 = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        # Time sampling strategy for better coverage
        if step < num_steps // 2:
            # Focus on early times initially
            t = torch.rand(1, device=device).item() * 0.8
        else:
            # Full time range later
            t = torch.rand(1, device=device).item()
        
        # Train step
        loss = trainer.train_step(x_0, t)
        losses.append(loss)
        
        # Validation checks
        if step in validation_steps or step % 100 == 0:
            # Test convergence
            if hasattr(graph, 'get_convergence_metrics'):
                metrics = graph.get_convergence_metrics(x_0, t=0.9, max_t=1.0)
                convergence_metrics.append({
                    'step': step,
                    't': 0.9,
                    **metrics
                })
                
                print(f"Step {step}: loss={loss:.4f}, KL@t=0.9={metrics['kl_from_uniform']:.4f}, "
                      f"entropy@t=0.9={metrics['entropy']:.4f}")
            else:
                print(f"Step {step}: loss={loss:.4f}")
    
    training_time = time.time() - start_time
    
    # Final validation
    print(f"\nRunning final validation for {experiment_name}...")
    validator = DiffusionValidator(vocab_size, device)
    
    # Test convergence
    convergence_results = validator.test_forward_diffusion_convergence(
        graph, noise, num_steps=50, batch_size=16, seq_len=32
    )
    
    # Test information decay
    decay_results = validator.test_information_decay(
        graph, noise, num_steps=50, num_sequences=10
    )
    
    # Compile results
    results = {
        'experiment_name': experiment_name,
        'training_time': training_time,
        'final_loss': np.mean(losses[-100:]),
        'losses': losses,
        'convergence_metrics': convergence_metrics,
        'validation': {
            'convergence': convergence_results['convergence'],
            'final_entropy': convergence_results['metrics']['entropy'][-1],
            'final_kl': convergence_results['metrics']['kl_from_uniform'][-1],
            'decay_rate': decay_results['decay_rate']
        }
    }
    
    return results


def compare_configurations(device='cuda'):
    """Compare different AEGUD configurations."""
    
    # Configuration
    vocab_size = 100
    hidden_dim = 128
    num_layers = 4
    num_steps = 1000
    
    # Noise schedule
    noise = GeometricNoise(sigma_min=0.001, sigma_max=10.0)
    
    experiments = {}
    
    # 1. Original AEGUD (baseline)
    print("\n" + "="*80)
    print("CONFIGURATION 1: Original AEGUD (Baseline)")
    print("="*80)
    
    model1 = create_simple_model(vocab_size, hidden_dim, num_layers, device)
    graph1 = AdaptiveUniform(
        dim=vocab_size,
        entropy_scale=1.0,
        sparsity_k=100
    ).to(device)
    
    results1 = run_experiment(
        "Original AEGUD",
        graph1, model1, noise,
        num_steps=num_steps,
        vocab_size=vocab_size,
        device=device
    )
    experiments['original'] = results1
    
    # 2. Enhanced AEGUD with Asymptotic Guarantee
    print("\n" + "="*80)
    print("CONFIGURATION 2: Enhanced AEGUD with Asymptotic Guarantee")
    print("="*80)
    
    model2 = create_simple_model(vocab_size, hidden_dim, num_layers, device)
    graph2 = create_enhanced_aegud(
        vocab_size,
        entropy_scale=1.0,
        sparsity_k=100,
        use_asymptotic_guarantee=True,
        use_two_stage=False,
        use_controlled_decay=False,
        kl_regularization_weight=0.0
    ).to(device)
    
    results2 = run_experiment(
        "AEGUD + Asymptotic",
        graph2, model2, noise,
        num_steps=num_steps,
        vocab_size=vocab_size,
        device=device
    )
    experiments['asymptotic'] = results2
    
    # 3. Enhanced AEGUD with Two-Stage Diffusion
    print("\n" + "="*80)
    print("CONFIGURATION 3: Enhanced AEGUD with Two-Stage Diffusion")
    print("="*80)
    
    model3 = create_simple_model(vocab_size, hidden_dim, num_layers, device)
    graph3 = create_enhanced_aegud(
        vocab_size,
        entropy_scale=1.0,
        sparsity_k=100,
        use_asymptotic_guarantee=False,
        use_two_stage=True,
        stage_transition_point=0.8,
        use_controlled_decay=False,
        kl_regularization_weight=0.0
    ).to(device)
    
    results3 = run_experiment(
        "AEGUD + Two-Stage",
        graph3, model3, noise,
        num_steps=num_steps,
        vocab_size=vocab_size,
        device=device
    )
    experiments['two_stage'] = results3
    
    # 4. Enhanced AEGUD with Controlled Decay
    print("\n" + "="*80)
    print("CONFIGURATION 4: Enhanced AEGUD with Controlled Information Decay")
    print("="*80)
    
    model4 = create_simple_model(vocab_size, hidden_dim, num_layers, device)
    graph4 = create_enhanced_aegud(
        vocab_size,
        entropy_scale=1.0,
        sparsity_k=100,
        use_asymptotic_guarantee=False,
        use_two_stage=False,
        use_controlled_decay=True,
        decay_tau=0.1,
        kl_regularization_weight=0.0
    ).to(device)
    
    results4 = run_experiment(
        "AEGUD + Controlled Decay",
        graph4, model4, noise,
        num_steps=num_steps,
        vocab_size=vocab_size,
        device=device
    )
    experiments['controlled_decay'] = results4
    
    # 5. Enhanced AEGUD with KL Regularization
    print("\n" + "="*80)
    print("CONFIGURATION 5: Enhanced AEGUD with KL Regularization")
    print("="*80)
    
    model5 = create_simple_model(vocab_size, hidden_dim, num_layers, device)
    graph5 = create_enhanced_aegud(
        vocab_size,
        entropy_scale=1.0,
        sparsity_k=100,
        use_asymptotic_guarantee=False,
        use_two_stage=False,
        use_controlled_decay=False,
        kl_regularization_weight=0.1
    ).to(device)
    
    results5 = run_experiment(
        "AEGUD + KL Regularization",
        graph5, model5, noise,
        num_steps=num_steps,
        vocab_size=vocab_size,
        device=device
    )
    experiments['kl_regularization'] = results5
    
    # 6. Enhanced AEGUD with All Features
    print("\n" + "="*80)
    print("CONFIGURATION 6: Enhanced AEGUD with All Features Combined")
    print("="*80)
    
    model6 = create_simple_model(vocab_size, hidden_dim, num_layers, device)
    graph6 = create_enhanced_aegud(
        vocab_size,
        entropy_scale=1.0,
        sparsity_k=100,
        use_asymptotic_guarantee=True,
        use_two_stage=True,
        stage_transition_point=0.8,
        use_controlled_decay=True,
        decay_tau=0.1,
        kl_regularization_weight=0.05
    ).to(device)
    
    results6 = run_experiment(
        "AEGUD + All Features",
        graph6, model6, noise,
        num_steps=num_steps,
        vocab_size=vocab_size,
        device=device
    )
    experiments['all_features'] = results6
    
    # 7. Baseline comparisons
    print("\n" + "="*80)
    print("BASELINE COMPARISONS")
    print("="*80)
    
    # Standard Uniform
    model_uniform = create_simple_model(vocab_size, hidden_dim, num_layers, device)
    graph_uniform = Uniform(vocab_size)
    
    results_uniform = run_experiment(
        "Standard Uniform",
        graph_uniform, model_uniform, noise,
        num_steps=num_steps,
        vocab_size=vocab_size,
        device=device
    )
    experiments['baseline_uniform'] = results_uniform
    
    # Standard Absorbing
    # Create special config for absorbing model with scale_by_sigma
    config_absorb = {
        'model': {
            'hidden_size': hidden_dim,
            'n_heads': 4,
            'n_blocks': num_layers,
            'cond_dim': 128,
            'dropout': 0.0,
            'scale_by_sigma': True  # Important for absorbing
        },
        'tokens': vocab_size,
        'graph': {
            'type': 'absorb'
        }
    }
    model_absorb = SEDD(config_absorb)
    graph_absorb = Absorbing(vocab_size)
    noise_absorb = LogLinearNoise()
    
    results_absorb = run_experiment(
        "Standard Absorbing",
        graph_absorb, model_absorb, noise_absorb,
        num_steps=num_steps,
        vocab_size=vocab_size,
        device=device
    )
    experiments['baseline_absorb'] = results_absorb
    
    return experiments


def create_comparison_plots(results, save_dir):
    """Create comparison visualizations."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data for plotting
    configs = list(results.keys())
    final_losses = [results[c]['final_loss'] for c in configs]
    final_kls = [results[c]['validation']['final_kl'] for c in configs]
    final_entropies = [results[c]['validation']['final_entropy'] for c in configs]
    convergence_status = [results[c]['validation']['convergence']['converged'] for c in configs]
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Final losses
    ax = axes[0, 0]
    colors = ['red' if not conv else 'green' for conv in convergence_status]
    bars = ax.bar(range(len(configs)), final_losses, color=colors, alpha=0.7)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_ylabel('Final Loss')
    ax.set_title('Final Training Loss by Configuration')
    ax.grid(True, alpha=0.3)
    
    # Add convergence indicators
    for i, (bar, conv) in enumerate(zip(bars, convergence_status)):
        if conv:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   '✓', ha='center', color='green', fontsize=12)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   '✗', ha='center', color='red', fontsize=12)
    
    # Plot 2: Final KL divergence
    ax = axes[0, 1]
    bars = ax.bar(range(len(configs)), final_kls, color=colors, alpha=0.7)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_ylabel('KL Divergence')
    ax.set_title('Final KL Divergence from Uniform')
    ax.axhline(y=0.01, color='red', linestyle='--', label='Target threshold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Final entropy
    ax = axes[1, 0]
    bars = ax.bar(range(len(configs)), final_entropies, color=colors, alpha=0.7)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_ylabel('Entropy')
    ax.set_title('Final Entropy (Normalized)')
    ax.axhline(y=0.95, color='red', linestyle='--', label='Target threshold')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary data
    table_data = []
    for config in configs:
        conv_symbol = '✓' if results[config]['validation']['convergence']['converged'] else '✗'
        table_data.append([
            config,
            f"{results[config]['final_loss']:.2f}",
            f"{results[config]['validation']['final_kl']:.4f}",
            f"{results[config]['validation']['final_entropy']:.3f}",
            conv_symbol
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Config', 'Loss', 'KL Div', 'Entropy', 'Conv.'],
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color code the convergence column
    for i in range(len(table_data)):
        if table_data[i][4] == '✓':
            table[(i+1, 4)].set_facecolor('lightgreen')
        else:
            table[(i+1, 4)].set_facecolor('lightcoral')
    
    plt.suptitle('Enhanced AEGUD Comparison Results', fontsize=16)
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"enhanced_comparison_{timestamp}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to {save_path}")
    plt.close()
    
    # Create loss curves plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for config in configs:
        losses = results[config]['losses']
        # Smooth losses for better visualization
        window = 50
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=config, alpha=0.7)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curves')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    loss_path = save_dir / f"loss_curves_{timestamp}.png"
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    print(f"Saved loss curves to {loss_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run enhanced AEGUD experiments')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for training')
    parser.add_argument('--num_steps', type=int, default=1000,
                       help='Number of training steps')
    parser.add_argument('--save_dir', type=str, 
                       default='experiments/aegud/results/enhanced',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    print(f"Running enhanced AEGUD experiments on {args.device}")
    print(f"Number of training steps: {args.num_steps}")
    
    # Run experiments
    results = compare_configurations(device=args.device)
    
    # Save results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = save_dir / f"enhanced_results_{timestamp}.json"
    
    # Convert numpy values to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nSaved results to {results_path}")
    
    # Create comparison plots
    create_comparison_plots(results, save_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    for config, result in results.items():
        conv_status = "CONVERGED" if result['validation']['convergence']['converged'] else "NOT CONVERGED"
        print(f"\n{config}:")
        print(f"  - Final Loss: {result['final_loss']:.4f}")
        print(f"  - Final KL: {result['validation']['final_kl']:.6f}")
        print(f"  - Final Entropy: {result['validation']['final_entropy']:.4f}")
        print(f"  - Convergence: {conv_status}")
        print(f"  - Information Decay Rate: {result['validation']['decay_rate']:.4f}")
    
    # Find best configuration
    converged_configs = [c for c, r in results.items() 
                        if r['validation']['convergence']['converged']]
    
    if converged_configs:
        best_config = min(converged_configs, key=lambda c: results[c]['final_loss'])
        print(f"\nBest converged configuration: {best_config}")
        print(f"  - Loss: {results[best_config]['final_loss']:.4f}")
        print(f"  - KL: {results[best_config]['validation']['final_kl']:.6f}")
    else:
        print("\nWARNING: No configurations achieved proper convergence!")


if __name__ == "__main__":
    main()