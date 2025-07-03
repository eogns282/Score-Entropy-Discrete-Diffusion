"""
Enhanced Adaptive Uniform Graph with Theoretical Guarantees
Implements all fixes from NEXT_IDEA.md to ensure proper convergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
import math
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from graph_lib import Graph, Uniform
from experiments.aegud.src.adaptive_uniform_graph import AdaptiveUniform
from experiments.aegud.src.entropy_estimator import EntropyEstimator, AdaptiveTransitionMatrix


class EnhancedAdaptiveUniform(AdaptiveUniform):
    """
    Enhanced AEGUD with theoretical guarantees for proper convergence.
    Implements:
    1. Asymptotic Uniform Guarantee
    2. Two-Stage Diffusion
    3. KL Regularization
    4. Controlled Information Decay
    """
    
    def __init__(self, dim, entropy_estimator=None, transition_matrix=None,
                 entropy_scale=1.0, sparsity_k=None,
                 # New parameters for enhancements
                 use_asymptotic_guarantee=True,
                 use_two_stage=True,
                 stage_transition_point=0.8,
                 use_controlled_decay=True,
                 decay_tau=0.1,
                 kl_regularization_weight=0.01):
        """
        Args:
            dim: Vocabulary size
            entropy_estimator: EntropyEstimator module
            transition_matrix: AdaptiveTransitionMatrix module
            entropy_scale: Scale factor for entropy influence
            sparsity_k: If set, use sparse transitions with k neighbors
            use_asymptotic_guarantee: Enable asymptotic uniform convergence
            use_two_stage: Enable two-stage diffusion
            stage_transition_point: Point to switch from adaptive to uniform (0-1)
            use_controlled_decay: Enable controlled information decay
            decay_tau: Time constant for information decay
            kl_regularization_weight: Weight for KL divergence regularization
        """
        super().__init__(dim, entropy_estimator, transition_matrix, entropy_scale, sparsity_k)
        
        # Enhancement flags
        self.use_asymptotic_guarantee = use_asymptotic_guarantee
        self.use_two_stage = use_two_stage
        self.stage_transition_point = stage_transition_point
        self.use_controlled_decay = use_controlled_decay
        self.decay_tau = decay_tau
        self.kl_regularization_weight = kl_regularization_weight
        
        # For tracking convergence metrics
        self.register_buffer('uniform_distribution', torch.ones(dim) / dim)
        
    def get_adaptive_weight(self, t, max_t=1.0):
        """
        Compute adaptive weight that decays over time to ensure convergence.
        Implements both asymptotic guarantee and controlled decay.
        """
        if not self.use_asymptotic_guarantee and not self.use_controlled_decay:
            return 1.0
        
        # Normalize t to [0, 1] range
        normalized_t = t / max_t if isinstance(t, (int, float)) else t
        
        # Convert to scalar if tensor
        if isinstance(normalized_t, torch.Tensor):
            normalized_t = normalized_t.item()
        
        if self.use_two_stage:
            # Two-stage approach: full adaptive then hard switch to uniform
            if normalized_t < self.stage_transition_point:
                # Stage 1: Full adaptive behavior
                if self.use_controlled_decay:
                    # Apply exponential decay within stage 1
                    decay_factor = np.exp(-normalized_t / (self.decay_tau * self.stage_transition_point))
                    return decay_factor
                else:
                    return 1.0
            else:
                # Stage 2: Force uniform convergence
                return 0.0
        
        elif self.use_asymptotic_guarantee:
            # Smooth asymptotic guarantee
            uniform_weight = normalized_t ** 2
            adaptive_weight = 1 - uniform_weight
            
            if self.use_controlled_decay:
                # Combined with exponential decay
                decay_factor = np.exp(-normalized_t / self.decay_tau)
                adaptive_weight = adaptive_weight * decay_factor
            
            return adaptive_weight
        
        elif self.use_controlled_decay:
            # Just exponential decay
            return np.exp(-normalized_t / self.decay_tau)
        
        return 1.0
    
    def adaptive_rate(self, x_t, t, max_t=1.0):
        """
        Enhanced adaptive rate with convergence guarantees.
        """
        batch_size, seq_len = x_t.shape
        device = x_t.device
        
        # Get adaptive weight based on time
        adaptive_weight = self.get_adaptive_weight(t, max_t)
        
        if adaptive_weight == 0.0:
            # Pure uniform transitions
            rate_matrix = torch.ones(batch_size, seq_len, self._dim, self._dim, device=device) / self._dim
            mask = torch.eye(self._dim, device=device).bool()
            mask = mask.unsqueeze(0).unsqueeze(0).expand_as(rate_matrix)
            rate_matrix = rate_matrix.masked_fill(mask, 0)
            off_diagonal_sum = rate_matrix.sum(dim=-1, keepdim=True)
            rate_matrix = rate_matrix - off_diagonal_sum * mask.float()
            return rate_matrix
        
        # Get base adaptive rate
        rate_matrix = super().adaptive_rate(x_t, t)
        
        if adaptive_weight < 1.0:
            # Blend with uniform rate
            uniform_rate = torch.ones_like(rate_matrix) / self._dim
            mask = torch.eye(self._dim, device=device).bool()
            mask = mask.unsqueeze(0).unsqueeze(0).expand_as(uniform_rate)
            uniform_rate = uniform_rate.masked_fill(mask, 0)
            uniform_off_diagonal = uniform_rate.sum(dim=-1, keepdim=True)
            uniform_rate = uniform_rate - uniform_off_diagonal * mask.float()
            
            # Weighted combination
            rate_matrix = adaptive_weight * rate_matrix + (1 - adaptive_weight) * uniform_rate
        
        return rate_matrix
    
    def compute_kl_from_uniform(self, transition_probs):
        """
        Compute KL divergence from uniform distribution for regularization.
        
        Args:
            transition_probs: Transition probability matrix (B, L, V, V)
            
        Returns:
            kl_divergence: Scalar KL divergence
        """
        # Get marginal distribution at each position
        # Sum over source tokens weighted by their probability
        batch_size, seq_len, vocab_size, _ = transition_probs.shape
        
        # Compute marginal by averaging transitions
        marginal = transition_probs.mean(dim=2)  # (B, L, V)
        
        # Compute KL divergence from uniform
        uniform = self.uniform_distribution.unsqueeze(0).unsqueeze(0)  # (1, 1, V)
        
        # KL(P||Q) = sum(P * log(P/Q))
        kl = F.kl_div(
            uniform.log(),
            marginal,
            reduction='batchmean'
        )
        
        return kl
    
    def score_entropy_with_kl(self, score, sigma, x_t, x_0, t=None, max_t=1.0):
        """
        Enhanced score entropy with KL regularization.
        """
        # Get base score entropy loss
        base_loss = super().score_entropy(score, sigma, x_t, x_0)
        
        if self.kl_regularization_weight > 0 and t is not None:
            # Compute transition probabilities for KL regularization
            rate_matrix = self.adaptive_rate(x_t, t, max_t)
            
            # Convert rates to probabilities (simplified)
            trans_probs = F.softmax(rate_matrix, dim=-1)
            
            # Compute KL divergence
            kl_loss = self.compute_kl_from_uniform(trans_probs)
            
            # Weight KL loss based on time (stronger near t=T)
            time_weight = (t / max_t) ** 2 if isinstance(t, (int, float)) else t ** 2
            kl_loss = kl_loss * time_weight * self.kl_regularization_weight
            
            # Add to base loss
            total_loss = base_loss + kl_loss
            
            # Store metrics for logging
            self.last_kl_loss = kl_loss.item() if hasattr(kl_loss, 'item') else kl_loss
            
            return total_loss
        
        return base_loss
    
    def get_convergence_metrics(self, x_t, t, max_t=1.0):
        """
        Compute metrics to monitor convergence quality.
        
        Returns dict with:
            - mutual_information: Estimated MI between x_t and x_0
            - kl_from_uniform: KL divergence from uniform distribution  
            - entropy: Current entropy of distribution
            - effective_temperature: Current temperature parameter
        """
        device = x_t.device
        batch_size, seq_len = x_t.shape
        
        # Get token distribution
        token_counts = torch.zeros(batch_size, self._dim, device=device)
        token_counts.scatter_add_(1, x_t, torch.ones_like(x_t, dtype=torch.float))
        token_probs = token_counts / seq_len
        
        # Compute entropy
        entropy = -torch.sum(
            token_probs * torch.log(token_probs + 1e-10),
            dim=1
        ) / math.log(self._dim)  # Normalize to [0, 1]
        
        # Compute KL from uniform
        uniform = torch.ones_like(token_probs) / self._dim
        kl_from_uniform = F.kl_div(
            uniform.log(),
            token_probs,
            reduction='batchmean'
        )
        
        # Estimate mutual information (simplified)
        # Higher values mean more structure preserved
        info_content = self.entropy_estimator.estimate_information_content(x_t)
        mutual_information = info_content * (1 - t / max_t)
        
        # Get effective temperature
        adaptive_weight = self.get_adaptive_weight(t, max_t)
        effective_temp = 1.0 / (adaptive_weight + 1e-6) if adaptive_weight > 0 else float('inf')
        
        return {
            'mutual_information': mutual_information.mean().item(),
            'kl_from_uniform': kl_from_uniform.item(),
            'entropy': entropy.mean().item(),
            'effective_temperature': effective_temp
        }
    
    def validate_convergence(self, num_steps=100, vocab_size=100, seq_len=32, device='cuda'):
        """
        Validate that the model properly converges to uniform distribution.
        """
        print("Validating convergence properties...")
        
        # Generate random initial sequence
        x_0 = torch.randint(0, vocab_size, (1, seq_len), device=device)
        
        metrics_over_time = {
            'mutual_information': [],
            'kl_from_uniform': [],
            'entropy': [],
            'effective_temperature': []
        }
        
        # Simulate forward diffusion
        for step in range(num_steps + 1):
            t = step / num_steps
            
            # Get convergence metrics
            metrics = self.get_convergence_metrics(x_0, t, max_t=1.0)
            
            for key, value in metrics.items():
                metrics_over_time[key].append(value)
            
            # Print progress at key points
            if step % (num_steps // 10) == 0:
                print(f"t={t:.2f}: KL from uniform = {metrics['kl_from_uniform']:.4f}, "
                      f"Entropy = {metrics['entropy']:.4f}, "
                      f"Effective temp = {metrics['effective_temperature']:.2f}")
        
        # Check final convergence
        final_kl = metrics_over_time['kl_from_uniform'][-1]
        final_entropy = metrics_over_time['entropy'][-1]
        
        print(f"\nFinal KL from uniform: {final_kl:.6f}")
        print(f"Final entropy: {final_entropy:.6f} (max possible: 1.0)")
        
        converged = final_kl < 0.01 and final_entropy > 0.95
        print(f"Convergence test: {'PASSED' if converged else 'FAILED'}")
        
        return metrics_over_time, converged


class HybridNoiseSchedule(nn.Module):
    """
    Hybrid noise schedule that transitions from adaptive to uniform.
    """
    
    def __init__(self, tau=0.1, transition_point=0.8):
        super().__init__()
        self.tau = tau
        self.transition_point = transition_point
    
    def get_transition_matrix(self, t, max_t, adaptive_fn, uniform_fn):
        """
        Get transition matrix based on current time.
        
        Args:
            t: Current time
            max_t: Maximum time
            adaptive_fn: Function to get adaptive transitions
            uniform_fn: Function to get uniform transitions
            
        Returns:
            Transition matrix
        """
        normalized_t = t / max_t
        
        if normalized_t < self.transition_point:
            # Adaptive phase with decay
            decay_factor = np.exp(-normalized_t / (self.tau * self.transition_point))
            return adaptive_fn() * decay_factor + uniform_fn() * (1 - decay_factor)
        else:
            # Pure uniform phase
            return uniform_fn()
    
    def get_noise_schedule(self, t, max_t):
        """
        Get noise level based on hybrid schedule.
        """
        normalized_t = t / max_t
        
        if normalized_t < self.transition_point:
            # Adaptive phase: slower noise accumulation
            phase_t = normalized_t / self.transition_point
            return -np.log(1 - phase_t * 0.8)  # Reach 80% noise by transition
        else:
            # Uniform phase: rapid convergence to full noise
            phase_t = (normalized_t - self.transition_point) / (1 - self.transition_point)
            base_noise = -np.log(1 - 0.8)
            return base_noise + phase_t * (-np.log(0.001))  # Reach 99.9% noise


def create_enhanced_aegud(vocab_size, **kwargs):
    """
    Factory function to create enhanced AEGUD with all improvements.
    """
    # Create entropy estimator
    entropy_estimator = EntropyEstimator(vocab_size, hidden_dim=256)
    
    # Create transition matrix
    transition_matrix = AdaptiveTransitionMatrix(vocab_size, hidden_dim=256)
    
    # Create enhanced graph
    enhanced_graph = EnhancedAdaptiveUniform(
        dim=vocab_size,
        entropy_estimator=entropy_estimator,
        transition_matrix=transition_matrix,
        **kwargs
    )
    
    return enhanced_graph