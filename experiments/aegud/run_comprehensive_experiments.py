"""
Comprehensive Experiments for Enhanced AEGUD V2
Tests all new research directions and creates detailed comparisons
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import SEDD components
from model.transformer import SEDD
from noise_lib import GeometricNoise, LogLinearNoise
from graph_lib import Uniform

# Import all AEGUD components
from experiments.aegud.src.adaptive_uniform_graph import AdaptiveUniform
from experiments.aegud.src.enhanced_adaptive_uniform_graph import EnhancedAdaptiveUniform
from experiments.aegud.src.enhanced_adaptive_uniform_graph_v2 import EnhancedAdaptiveUniformV2
from experiments.aegud.src.hierarchical_diffusion import HierarchicalDiffusion
from experiments.aegud.src.enhanced_metrics import EnhancedMetrics
from experiments.aegud.src.entropy_estimator import EntropyEstimator, AdaptiveTransitionMatrix
from experiments.aegud.src.simple_loss import SimpleLoss
from experiments.aegud.src.diffusion_validator import DiffusionValidator

# Set random seed
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class ComprehensiveExperimentRunner:
    """Run and compare all AEGUD variants comprehensively."""
    
    def __init__(self, base_config, device='cuda'):
        self.config = base_config
        self.device = device
        self.results = {}
        self.metrics_tracker = EnhancedMetrics(base_config['vocab_size'], device)
        
    def create_graph_variants(self):
        """Create all graph variants to test."""
        vocab_size = self.config['vocab_size']
        
        variants = {
            # Baseline
            'baseline_uniform': Uniform(vocab_size),
            
            # Original AEGUD
            'original_aegud': AdaptiveUniform(
                dim=vocab_size,
                entropy_scale=1.0
            ),
            
            # Enhanced AEGUD (original fixes)
            'enhanced_aegud_v1': EnhancedAdaptiveUniform(
                dim=vocab_size,
                entropy_scale=1.0,
                use_asymptotic_guarantee=True,
                use_two_stage=True,
                stage_transition_point=0.8,
                use_controlled_decay=True,
                decay_tau=0.1,
                kl_regularization_weight=0.01
            ),
            
            # Enhanced AEGUD V2 - Vocabulary Aware
            'enhanced_v2_vocab_aware': EnhancedAdaptiveUniformV2(
                dim=vocab_size,
                entropy_scale=1.0,
                use_vocabulary_aware_decay=True,
                use_learnable_schedule=False,
                use_information_bottleneck=False,
                use_two_stage=True,
                stage_transition_point=0.8
            ),
            
            # Enhanced AEGUD V2 - Learnable Schedule
            'enhanced_v2_learnable': EnhancedAdaptiveUniformV2(
                dim=vocab_size,
                entropy_scale=1.0,
                use_vocabulary_aware_decay=False,
                use_learnable_schedule=True,
                use_information_bottleneck=False
            ),
            
            # Enhanced AEGUD V2 - Information Bottleneck
            'enhanced_v2_info_bottleneck': EnhancedAdaptiveUniformV2(
                dim=vocab_size,
                entropy_scale=1.0,
                use_vocabulary_aware_decay=False,
                use_learnable_schedule=False,
                use_information_bottleneck=True,
                info_bottleneck_beta=0.1
            ),
            
            # Enhanced AEGUD V2 - Full Features
            'enhanced_v2_full': EnhancedAdaptiveUniformV2(
                dim=vocab_size,
                entropy_scale=1.0,
                use_vocabulary_aware_decay=True,
                use_learnable_schedule=True,
                use_information_bottleneck=True,
                use_two_stage=True,
                stage_transition_point=0.8,
                use_controlled_decay=True,
                decay_tau=0.1,
                kl_regularization_weight=0.05,
                relaxed_convergence_epsilon=0.1
            ),
            
            # Enhanced AEGUD V2 - Relaxed Convergence
            'enhanced_v2_relaxed': EnhancedAdaptiveUniformV2(
                dim=vocab_size,
                entropy_scale=1.0,
                use_two_stage=True,
                relaxed_convergence_epsilon=0.2  # More relaxed
            )
        }
        
        # Move all to device
        for name, graph in variants.items():
            if hasattr(graph, 'to'):
                variants[name] = graph.to(self.device)
        
        return variants
    
    def test_convergence_properties(self, graph_name, graph, noise):
        """Test convergence properties of a graph."""
        print(f"\nTesting convergence for {graph_name}...")
        
        # Run convergence test
        convergence_metrics = self.metrics_tracker.measure_convergence_quality(
            graph, noise,
            num_steps=self.config['convergence_steps'],
            batch_size=self.config['batch_size'],
            seq_len=self.config['seq_len']
        )
        
        # Create visualization
        viz_path = f"experiments/aegud/visualizations/convergence_{graph_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        os.makedirs(os.path.dirname(viz_path), exist_ok=True)
        self.metrics_tracker.create_convergence_visualization(convergence_metrics, viz_path)
        
        return convergence_metrics
    
    def test_information_preservation(self, graph_name, graph, noise):
        """Test information preservation during diffusion."""
        print(f"\nTesting information preservation for {graph_name}...")
        
        batch_size = self.config['batch_size']
        seq_len = self.config['seq_len']
        vocab_size = self.config['vocab_size']
        
        # Generate test sequences
        x_0 = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        
        preservation_metrics = []
        time_steps = np.linspace(0, 1, 21)
        
        for t in time_steps:
            # Get noise
            sigma, _ = noise(torch.tensor([t], device=self.device))
            
            # Apply diffusion
            x_t = graph.sample_transition(x_0, sigma.expand(batch_size, 1))
            
            # Measure preservation
            metrics = self.metrics_tracker.measure_semantic_preservation(x_0, x_t, t)
            metrics['time'] = t
            preservation_metrics.append(metrics)
        
        return preservation_metrics
    
    def test_generation_diversity(self, graph_name, graph, noise, model=None):
        """Test diversity of generated samples."""
        print(f"\nTesting generation diversity for {graph_name}...")
        
        # For now, we'll test diversity of forward diffusion samples
        # In full implementation, would use reverse diffusion with model
        
        samples = []
        for _ in range(10):
            x_0 = torch.randint(0, self.config['vocab_size'], 
                              (self.config['batch_size'], self.config['seq_len']), 
                              device=self.device)
            
            # Apply varying amounts of noise
            for t in [0.3, 0.5, 0.7, 0.9]:
                sigma, _ = noise(torch.tensor([t], device=self.device))
                x_t = graph.sample_transition(x_0, sigma.expand(x_0.shape[0], 1))
                samples.append(x_t)
        
        diversity_metrics = self.metrics_tracker.analyze_generation_diversity(samples)
        
        return diversity_metrics
    
    def run_quick_training_test(self, graph_name, graph, noise):
        """Run a quick training test to check stability."""
        print(f"\nRunning quick training test for {graph_name}...")
        
        # Create simple model
        model_config = {
            'model': {
                'hidden_size': 128,
                'n_heads': 4,
                'n_blocks': 4,
                'cond_dim': 64,
                'dropout': 0.1,
                'scale_by_sigma': False
            },
            'tokens': self.config['vocab_size'],
            'graph': {'type': 'uniform'}
        }
        
        model = SEDD(model_config).to(self.device)
        
        # Create optimizer
        params = list(model.parameters())
        if hasattr(graph, 'parameters'):
            params.extend(graph.parameters())
        
        optimizer = torch.optim.AdamW(params, lr=1e-3)
        
        # Create loss function
        if hasattr(graph, 'score_entropy_with_kl'):
            def loss_fn(score, x_t, x_0, sigma, graph, t=None):
                return graph.score_entropy_with_kl(score, sigma, x_t, x_0, t=t, max_t=1.0).mean()
        else:
            loss_fn = SimpleLoss()
        
        # Training loop
        losses = []
        model.train()
        
        for step in range(self.config['quick_train_steps']):
            # Generate batch
            x_0 = torch.randint(0, self.config['vocab_size'], 
                              (self.config['batch_size'], self.config['seq_len']), 
                              device=self.device)
            
            # Sample time
            t = torch.rand(self.config['batch_size'], device=self.device)
            sigma, dsigma = noise(t)
            
            # Forward diffusion
            x_t = graph.sample_transition(x_0, sigma[:, None])
            
            # Model forward
            score = model(x_t, sigma)
            
            # Compute loss
            if hasattr(graph, 'score_entropy_with_kl'):
                loss = loss_fn(score, x_t, x_0, sigma, graph, t=t.mean())
            else:
                loss = loss_fn(score, x_t, x_0, sigma, graph)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if step % 10 == 0:
                print(f"  Step {step}: loss={np.mean(losses[-10:]):.4f}")
        
        return {
            'final_loss': np.mean(losses[-10:]),
            'loss_trajectory': losses,
            'stable': np.std(losses[-50:]) < 1.0  # Check if training is stable
        }
    
    def run_all_experiments(self):
        """Run all experiments and comparisons."""
        print("="*80)
        print("Running Comprehensive AEGUD Experiments")
        print("="*80)
        
        # Create noise schedule
        noise = GeometricNoise(sigma_min=0.001, sigma_max=100.0)
        
        # Create all graph variants
        graph_variants = self.create_graph_variants()
        
        # Results storage
        all_results = {}
        
        for graph_name, graph in graph_variants.items():
            print(f"\n{'#'*60}")
            print(f"# Testing: {graph_name}")
            print(f"{'#'*60}")
            
            try:
                results = {}
                
                # 1. Convergence test
                convergence = self.test_convergence_properties(graph_name, graph, noise)
                results['convergence'] = {
                    'final_entropy': convergence['entropy_trajectory'][-1],
                    'final_kl': convergence['kl_trajectory'][-1],
                    'converged': convergence['final_status']['overall_converged'],
                    'convergence_rate': convergence['convergence_rate']
                }
                
                # 2. Information preservation test
                preservation = self.test_information_preservation(graph_name, graph, noise)
                results['preservation'] = {
                    'early_preservation': np.mean([m['token_overlap'] for m in preservation[:5]]),
                    'mid_preservation': np.mean([m['token_overlap'] for m in preservation[8:13]]),
                    'late_preservation': np.mean([m['token_overlap'] for m in preservation[16:]])
                }
                
                # 3. Diversity test
                diversity = self.test_generation_diversity(graph_name, graph, noise)
                results['diversity'] = diversity
                
                # 4. Quick training test
                training = self.run_quick_training_test(graph_name, graph, noise)
                results['training'] = {
                    'final_loss': training['final_loss'],
                    'stable': training['stable']
                }
                
                all_results[graph_name] = results
                
            except Exception as e:
                print(f"Error testing {graph_name}: {str(e)}")
                all_results[graph_name] = {'error': str(e)}
        
        self.results = all_results
        return all_results
    
    def create_comparison_report(self, save_dir="experiments/aegud/results"):
        """Create comprehensive comparison report."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert results to DataFrame for easier analysis
        rows = []
        for graph_name, results in self.results.items():
            if 'error' not in results:
                row = {
                    'Graph': graph_name,
                    'Final Entropy': results['convergence']['final_entropy'],
                    'Final KL': results['convergence']['final_kl'],
                    'Converged': results['convergence']['converged'],
                    'Conv. Rate': results['convergence']['convergence_rate'],
                    'Early Preservation': results['preservation']['early_preservation'],
                    'Token Diversity': results['diversity'].get('token_entropy', 0),
                    'Training Loss': results['training']['final_loss'],
                    'Stable Training': results['training']['stable']
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive AEGUD Comparison', fontsize=16)
        
        # 1. Convergence comparison
        ax = axes[0, 0]
        df_sorted = df.sort_values('Final KL')
        ax.barh(df_sorted['Graph'], df_sorted['Final KL'])
        ax.set_xlabel('Final KL Divergence')
        ax.set_title('Convergence Quality (Lower is Better)')
        ax.axvline(x=0.01, color='r', linestyle='--', alpha=0.5, label='Target')
        
        # 2. Information preservation
        ax = axes[0, 1]
        df_sorted = df.sort_values('Early Preservation', ascending=False)
        ax.barh(df_sorted['Graph'], df_sorted['Early Preservation'])
        ax.set_xlabel('Early Stage Preservation')
        ax.set_title('Information Preservation (Higher is Better)')
        
        # 3. Diversity comparison
        ax = axes[0, 2]
        df_sorted = df.sort_values('Token Diversity', ascending=False)
        ax.barh(df_sorted['Graph'], df_sorted['Token Diversity'])
        ax.set_xlabel('Token Entropy')
        ax.set_title('Generation Diversity (Higher is Better)')
        
        # 4. Training stability
        ax = axes[1, 0]
        stable_counts = df['Stable Training'].value_counts()
        ax.pie(stable_counts.values, labels=['Stable', 'Unstable'], autopct='%1.1f%%')
        ax.set_title('Training Stability')
        
        # 5. Overall scores (normalized)
        ax = axes[1, 1]
        # Normalize metrics to 0-1 scale
        df_norm = df.copy()
        df_norm['Conv Score'] = 1 - (df_norm['Final KL'] - df_norm['Final KL'].min()) / (df_norm['Final KL'].max() - df_norm['Final KL'].min())
        df_norm['Pres Score'] = (df_norm['Early Preservation'] - df_norm['Early Preservation'].min()) / (df_norm['Early Preservation'].max() - df_norm['Early Preservation'].min())
        df_norm['Div Score'] = (df_norm['Token Diversity'] - df_norm['Token Diversity'].min()) / (df_norm['Token Diversity'].max() - df_norm['Token Diversity'].min())
        df_norm['Overall Score'] = (df_norm['Conv Score'] + df_norm['Pres Score'] + df_norm['Div Score']) / 3
        
        df_sorted = df_norm.sort_values('Overall Score', ascending=False)
        ax.barh(df_sorted['Graph'], df_sorted['Overall Score'])
        ax.set_xlabel('Overall Score')
        ax.set_title('Combined Performance (Higher is Better)')
        
        # 6. Summary table
        ax = axes[1, 2]
        ax.axis('off')
        
        # Create summary text
        best_convergence = df.loc[df['Final KL'].idxmin(), 'Graph']
        best_preservation = df.loc[df['Early Preservation'].idxmax(), 'Graph']
        best_diversity = df.loc[df['Token Diversity'].idxmax(), 'Graph']
        best_overall = df_sorted.iloc[0]['Graph']
        
        summary_text = f"""
Best Performers:

Convergence: {best_convergence}
KL = {df.loc[df['Graph'] == best_convergence, 'Final KL'].values[0]:.4f}

Preservation: {best_preservation}
Score = {df.loc[df['Graph'] == best_preservation, 'Early Preservation'].values[0]:.4f}

Diversity: {best_diversity}
Entropy = {df.loc[df['Graph'] == best_diversity, 'Token Diversity'].values[0]:.4f}

Overall: {best_overall}
Score = {df_sorted.iloc[0]['Overall Score']:.4f}
"""
        
        ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_dir}/comprehensive_comparison_{timestamp}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save detailed results
        df.to_csv(f"{save_dir}/detailed_results_{timestamp}.csv", index=False)
        
        # Save raw results
        with open(f"{save_dir}/raw_results_{timestamp}.json", 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nResults saved to {save_dir}")
        
        return df


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive AEGUD experiments')
    parser.add_argument('--vocab_size', type=int, default=100, help='Vocabulary size')
    parser.add_argument('--seq_len', type=int, default=32, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--convergence_steps', type=int, default=50, help='Steps for convergence test')
    parser.add_argument('--quick_train_steps', type=int, default=100, help='Steps for quick training test')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Configuration
    config = {
        'vocab_size': args.vocab_size,
        'seq_len': args.seq_len,
        'batch_size': args.batch_size,
        'convergence_steps': args.convergence_steps,
        'quick_train_steps': args.quick_train_steps
    }
    
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Run experiments
    runner = ComprehensiveExperimentRunner(config, device=args.device)
    results = runner.run_all_experiments()
    
    # Create report
    df = runner.create_comparison_report()
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(df.to_string())
    
    # Test hierarchical diffusion separately
    print("\n" + "="*80)
    print("Testing Hierarchical Diffusion")
    print("="*80)
    
    hierarchical = HierarchicalDiffusion(args.vocab_size, num_levels=3)
    hierarchical = hierarchical.to(args.device)
    
    # Validate hierarchical convergence
    hierarchical_results = hierarchical.validate_hierarchical_convergence(
        num_steps=args.convergence_steps, 
        device=args.device
    )
    
    print("\nHierarchical diffusion tested successfully!")


if __name__ == "__main__":
    main()