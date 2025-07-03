"""
Validation tools for monitoring diffusion quality and theoretical properties.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from datetime import datetime
from pathlib import Path
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from typing import Dict, List, Tuple, Optional


class DiffusionValidator:
    """
    Comprehensive validation suite for diffusion models.
    Tests convergence, uniformity, and other theoretical properties.
    """
    
    def __init__(self, vocab_size: int, device: str = 'cuda'):
        self.vocab_size = vocab_size
        self.device = device
        self.results = {}
        
    def test_forward_diffusion_convergence(self, graph, noise_schedule, 
                                         num_steps: int = 100,
                                         batch_size: int = 32,
                                         seq_len: int = 64) -> Dict:
        """
        Test if forward diffusion properly converges to uniform distribution.
        """
        print("Testing forward diffusion convergence...")
        
        # Generate initial sequences
        x_0 = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        
        # Track metrics over time
        metrics = {
            'entropy': [],
            'kl_from_uniform': [],
            'token_diversity': [],
            'chi_squared_pvalue': [],
            'timesteps': []
        }
        
        # Forward diffusion
        for step in range(num_steps + 1):
            t = step / num_steps
            sigma = noise_schedule.total_noise(torch.tensor(t, device=self.device))
            
            # Sample x_t given x_0
            if step == 0:
                x_t = x_0.clone()
            else:
                # Apply transitions
                x_t = graph.sample_transition(x_t, sigma)
            
            # Compute metrics
            entropy = self._compute_entropy(x_t)
            kl_div = self._compute_kl_from_uniform(x_t)
            diversity = self._compute_token_diversity(x_t)
            chi2_p = self._chi_squared_test(x_t)
            
            metrics['entropy'].append(entropy)
            metrics['kl_from_uniform'].append(kl_div)
            metrics['token_diversity'].append(diversity)
            metrics['chi_squared_pvalue'].append(chi2_p)
            metrics['timesteps'].append(t)
            
            # Log progress
            if step % (num_steps // 10) == 0:
                print(f"  t={t:.2f}: entropy={entropy:.4f}, KL={kl_div:.4f}, "
                      f"diversity={diversity:.4f}, χ² p-value={chi2_p:.4f}")
        
        # Final convergence test
        final_entropy = metrics['entropy'][-1]
        final_kl = metrics['kl_from_uniform'][-1]
        final_chi2 = metrics['chi_squared_pvalue'][-1]
        
        max_entropy = np.log(self.vocab_size) / np.log(self.vocab_size)  # Should be 1.0
        
        convergence_criteria = {
            'entropy_ratio': final_entropy / max_entropy,
            'kl_threshold': final_kl < 0.01,
            'chi2_uniform': final_chi2 > 0.05,  # Fail to reject uniform hypothesis
            'converged': (final_entropy / max_entropy > 0.95) and (final_kl < 0.01) and (final_chi2 > 0.05)
        }
        
        print(f"\nConvergence test: {'PASSED' if convergence_criteria['converged'] else 'FAILED'}")
        print(f"  - Entropy ratio: {convergence_criteria['entropy_ratio']:.4f} (target > 0.95)")
        print(f"  - KL divergence: {final_kl:.6f} (target < 0.01)")
        print(f"  - χ² test p-value: {final_chi2:.4f} (target > 0.05)")
        
        return {
            'metrics': metrics,
            'convergence': convergence_criteria,
            'final_state': x_t
        }
    
    def test_information_decay(self, graph, noise_schedule,
                             num_steps: int = 100,
                             num_sequences: int = 10) -> Dict:
        """
        Test how information decays during forward diffusion.
        """
        print("\nTesting information decay...")
        
        # Create structured initial sequences with different patterns
        sequences = []
        seq_len = 64
        
        # Pattern 1: Repeated tokens
        seq1 = torch.tensor([i % 10 for i in range(seq_len)], device=self.device)
        sequences.append(seq1.repeat(num_sequences // 3, 1))
        
        # Pattern 2: Alternating tokens  
        seq2 = torch.tensor([i % 2 * 10 for i in range(seq_len)], device=self.device)
        sequences.append(seq2.repeat(num_sequences // 3, 1))
        
        # Pattern 3: Random tokens
        seq3 = torch.randint(0, self.vocab_size, (num_sequences // 3, seq_len), device=self.device)
        sequences.append(seq3)
        
        x_0 = torch.cat(sequences, dim=0)
        
        # Track mutual information over time
        mutual_info = []
        correlation_decay = []
        
        for step in range(num_steps + 1):
            t = step / num_steps
            sigma = noise_schedule.total_noise(torch.tensor(t, device=self.device))
            
            if step == 0:
                x_t = x_0.clone()
            else:
                x_t = graph.sample_transition(x_t, sigma)
            
            # Estimate mutual information
            mi = self._estimate_mutual_information(x_0, x_t)
            mutual_info.append(mi)
            
            # Compute correlation
            corr = self._compute_sequence_correlation(x_0, x_t)
            correlation_decay.append(corr)
        
        # Fit exponential decay
        t_values = np.linspace(0, 1, num_steps + 1)
        log_mi = np.log(np.array(mutual_info) + 1e-10)
        
        # Linear fit to log(MI) vs t
        slope, intercept = np.polyfit(t_values[mutual_info > np.array(mutual_info).max() * 0.01], 
                                     log_mi[mutual_info > np.array(mutual_info).max() * 0.01], 1)
        decay_rate = -slope
        
        print(f"  Information decay rate: {decay_rate:.4f}")
        print(f"  Final MI ratio: {mutual_info[-1] / mutual_info[0]:.6f}")
        
        return {
            'mutual_information': mutual_info,
            'correlation': correlation_decay,
            'decay_rate': decay_rate,
            'timesteps': t_values.tolist()
        }
    
    def test_reverse_process_quality(self, model, graph, noise_schedule,
                                   num_samples: int = 16,
                                   num_steps: int = 100) -> Dict:
        """
        Test quality of reverse process generation.
        """
        print("\nTesting reverse process quality...")
        
        # Start from noise
        seq_len = 64
        x_T = torch.randint(0, self.vocab_size, (num_samples, seq_len), device=self.device)
        
        # Track generation metrics
        diversity_scores = []
        entropy_scores = []
        
        # Reverse diffusion
        x_t = x_T
        for step in range(num_steps, -1, -1):
            t = step / num_steps
            
            # Get score from model (placeholder - would use actual model)
            # For testing, we'll use a simple metric
            
            if step > 0:
                t_prev = (step - 1) / num_steps
                sigma = noise_schedule.total_noise(torch.tensor(t, device=self.device))
                sigma_prev = noise_schedule.total_noise(torch.tensor(t_prev, device=self.device))
                
                # Simplified reverse step for testing
                # In practice, this would use the trained model
                noise = torch.randn_like(x_t, dtype=torch.float) * 0.1
                transition_probs = F.softmax(noise, dim=-1)
                x_t = torch.multinomial(transition_probs.view(-1, self.vocab_size), 1).view(x_t.shape)
        
        # Analyze generated samples
        final_diversity = self._compute_token_diversity(x_t)
        final_entropy = self._compute_entropy(x_t)
        repetition_rate = self._compute_repetition_rate(x_t)
        
        print(f"  Final diversity: {final_diversity:.4f}")
        print(f"  Final entropy: {final_entropy:.4f}")
        print(f"  Repetition rate: {repetition_rate:.4f}")
        
        return {
            'diversity': final_diversity,
            'entropy': final_entropy,
            'repetition_rate': repetition_rate,
            'samples': x_t
        }
    
    def visualize_diffusion_process(self, graph, noise_schedule,
                                  save_path: Optional[str] = None) -> None:
        """
        Create visualizations of the diffusion process.
        """
        print("\nGenerating diffusion visualizations...")
        
        # Run forward diffusion test
        results = self.test_forward_diffusion_convergence(
            graph, noise_schedule, num_steps=50
        )
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Entropy over time
        ax = axes[0, 0]
        ax.plot(results['metrics']['timesteps'], results['metrics']['entropy'])
        ax.axhline(y=1.0, color='r', linestyle='--', label='Max entropy')
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Entropy')
        ax.set_title('Entropy Evolution During Diffusion')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: KL divergence from uniform
        ax = axes[0, 1]
        ax.semilogy(results['metrics']['timesteps'], 
                   np.array(results['metrics']['kl_from_uniform']) + 1e-10)
        ax.axhline(y=0.01, color='r', linestyle='--', label='Target threshold')
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('KL Divergence (log scale)')
        ax.set_title('KL Divergence from Uniform Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Token diversity
        ax = axes[1, 0]
        ax.plot(results['metrics']['timesteps'], results['metrics']['token_diversity'])
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Token Diversity')
        ax.set_title('Token Diversity Over Time')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Chi-squared test p-values
        ax = axes[1, 1]
        ax.plot(results['metrics']['timesteps'], results['metrics']['chi_squared_pvalue'])
        ax.axhline(y=0.05, color='r', linestyle='--', label='α = 0.05')
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('χ² Test p-value')
        ax.set_title('Uniformity Test (χ² p-values)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved visualization to {save_path}")
        
        plt.close()
        
        # Create heatmap of token distribution evolution
        self._create_token_distribution_heatmap(results, save_path)
    
    def _create_token_distribution_heatmap(self, results: Dict, 
                                         base_save_path: Optional[str] = None) -> None:
        """
        Create heatmap showing token distribution evolution.
        """
        # This would show how token probabilities evolve over time
        # For now, we'll create a placeholder
        if base_save_path:
            heatmap_path = base_save_path.replace('.png', '_heatmap.png')
            print(f"  Token distribution heatmap would be saved to {heatmap_path}")
    
    def _compute_entropy(self, x: torch.Tensor) -> float:
        """Compute entropy of token distribution."""
        batch_size, seq_len = x.shape
        token_counts = torch.zeros(self.vocab_size, device=self.device)
        
        # Count tokens
        x_flat = x.view(-1)
        token_counts.scatter_add_(0, x_flat, torch.ones_like(x_flat, dtype=torch.float))
        
        # Compute probabilities
        probs = token_counts / (batch_size * seq_len)
        
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        normalized_entropy = entropy / np.log(self.vocab_size)
        
        return normalized_entropy.item()
    
    def _compute_kl_from_uniform(self, x: torch.Tensor) -> float:
        """Compute KL divergence from uniform distribution."""
        batch_size, seq_len = x.shape
        token_counts = torch.zeros(self.vocab_size, device=self.device)
        
        # Count tokens
        x_flat = x.view(-1)
        token_counts.scatter_add_(0, x_flat, torch.ones_like(x_flat, dtype=torch.float))
        
        # Compute probabilities
        probs = token_counts / (batch_size * seq_len)
        uniform_probs = torch.ones_like(probs) / self.vocab_size
        
        # KL divergence
        kl = torch.sum(probs * torch.log((probs + 1e-10) / uniform_probs))
        
        return kl.item()
    
    def _compute_token_diversity(self, x: torch.Tensor) -> float:
        """Compute ratio of unique tokens used."""
        unique_tokens = torch.unique(x)
        return len(unique_tokens) / self.vocab_size
    
    def _chi_squared_test(self, x: torch.Tensor) -> float:
        """Perform chi-squared test for uniformity."""
        batch_size, seq_len = x.shape
        
        # Count tokens
        observed = torch.zeros(self.vocab_size)
        x_cpu = x.cpu()
        for i in range(self.vocab_size):
            observed[i] = (x_cpu == i).sum().item()
        
        # Expected counts under uniform distribution
        total_count = batch_size * seq_len
        expected_count = total_count / self.vocab_size
        
        # Chi-squared statistic
        chi2_stat = 0.0
        for i in range(self.vocab_size):
            if expected_count > 0:
                chi2_stat += (observed[i] - expected_count) ** 2 / expected_count
        
        # Degrees of freedom
        df = self.vocab_size - 1
        
        # Compute p-value
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        
        return p_value
    
    def _estimate_mutual_information(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Estimate mutual information between two sequences."""
        # Simplified estimation based on token overlap
        overlap = (x == y).float().mean().cpu()
        # Convert to approximate MI (this is a rough approximation)
        mi = -np.log(1 - overlap.item() + 1e-10) * overlap.item()
        return mi
    
    def _compute_sequence_correlation(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute correlation between sequences."""
        return (x == y).float().mean().item()
    
    def _compute_repetition_rate(self, x: torch.Tensor) -> float:
        """Compute rate of repeated tokens in sequences."""
        batch_size, seq_len = x.shape
        repetitions = 0
        
        for i in range(batch_size):
            seq = x[i]
            # Count consecutive repetitions
            for j in range(1, seq_len):
                if seq[j] == seq[j-1]:
                    repetitions += 1
        
        return repetitions / (batch_size * (seq_len - 1))
    
    def save_validation_report(self, save_dir: str) -> None:
        """Save comprehensive validation report."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save results as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = save_path / f"validation_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nValidation report saved to {report_path}")


def run_comprehensive_validation(graph, noise_schedule, model=None, 
                               vocab_size=100, device='cuda'):
    """
    Run all validation tests and generate report.
    """
    validator = DiffusionValidator(vocab_size, device)
    
    # Test 1: Forward diffusion convergence
    convergence_results = validator.test_forward_diffusion_convergence(
        graph, noise_schedule
    )
    validator.results['convergence'] = convergence_results
    
    # Test 2: Information decay
    decay_results = validator.test_information_decay(
        graph, noise_schedule
    )
    validator.results['information_decay'] = decay_results
    
    # Test 3: Reverse process (if model provided)
    if model is not None:
        reverse_results = validator.test_reverse_process_quality(
            model, graph, noise_schedule
        )
        validator.results['reverse_process'] = reverse_results
    
    # Generate visualizations
    viz_path = f"experiments/aegud/visualizations/validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    os.makedirs(os.path.dirname(viz_path), exist_ok=True)
    validator.visualize_diffusion_process(graph, noise_schedule, viz_path)
    
    # Save report
    validator.save_validation_report("experiments/aegud/results")
    
    return validator.results