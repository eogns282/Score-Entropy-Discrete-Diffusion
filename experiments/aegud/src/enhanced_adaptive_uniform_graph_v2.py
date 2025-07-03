"""
Enhanced Adaptive Uniform Graph V2 - Fixed and Extended
Implements all fixes from NEW_NEXT_IDEA.md plus new research directions
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


class VocabularyAwareDecay(nn.Module):
    """
    Implements vocabulary-aware decay rates based on token characteristics.
    Different tokens decay at different rates based on their frequency and semantic importance.
    """
    
    def __init__(self, vocab_size, hidden_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Learn token-specific decay rates
        self.token_decay_net = nn.Sequential(
            nn.Embedding(vocab_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Frequency-based modulation (can be updated with real token frequencies)
        self.register_buffer('token_frequencies', torch.ones(vocab_size) / vocab_size)
        
    def forward(self, token_ids, t, base_decay_rate=0.1):
        """
        Compute vocabulary-aware decay for each token.
        
        Args:
            token_ids: Token indices (batch_size, seq_len)
            t: Current time
            base_decay_rate: Base decay rate
            
        Returns:
            decay_weights: Token-specific decay weights
        """
        # Get learned decay modulation for each token
        decay_modulation = self.token_decay_net(token_ids).squeeze(-1)
        
        # Get frequency-based modulation
        frequencies = self.token_frequencies[token_ids]
        freq_modulation = torch.log1p(frequencies * self.vocab_size)  # Higher freq = slower decay
        
        # Combine modulations
        combined_modulation = decay_modulation * freq_modulation
        
        # Apply exponential decay with token-specific rates
        decay_weights = torch.exp(-t / (base_decay_rate * (1 + combined_modulation)))
        
        return decay_weights
    
    def update_frequencies(self, token_counts):
        """Update token frequency statistics from corpus."""
        total_counts = token_counts.sum()
        self.token_frequencies = token_counts / total_counts


class LearnableConvergenceSchedule(nn.Module):
    """
    Learnable convergence schedule that adapts based on the data.
    The model learns when and how to transition from adaptive to uniform.
    """
    
    def __init__(self, hidden_dim=128):
        super().__init__()
        
        self.schedule_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Learnable transition point
        self.transition_point = nn.Parameter(torch.tensor(0.8))
        
    def forward(self, t, max_t=1.0):
        """
        Compute adaptive weight based on learned schedule.
        
        Args:
            t: Current time
            max_t: Maximum time
            
        Returns:
            adaptive_weight: Weight for adaptive behavior (0 = uniform, 1 = fully adaptive)
        """
        normalized_t = t / max_t if isinstance(t, (int, float)) else t
        
        if isinstance(normalized_t, torch.Tensor):
            t_input = normalized_t.view(-1, 1)
        else:
            t_input = torch.tensor([[normalized_t]], dtype=torch.float32)
        
        # Get learned schedule
        schedule_value = self.schedule_net(t_input).squeeze()
        
        # Apply smooth transition around learned transition point
        transition_smoothness = 0.1
        transition_factor = torch.sigmoid(
            (self.transition_point - normalized_t) / transition_smoothness
        )
        
        # Combine schedule with transition
        adaptive_weight = schedule_value * transition_factor
        
        return adaptive_weight.item() if adaptive_weight.numel() == 1 else adaptive_weight


class InformationBottleneck(nn.Module):
    """
    Information Bottleneck approach for optimal compression during diffusion.
    Maximizes I(X_t; Y) - β * I(X_t; X_0) where Y is target distribution.
    """
    
    def __init__(self, vocab_size, hidden_dim=256, beta=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.beta = beta
        
        # Encoder: X_0 -> Z (compressed representation)
        self.encoder = nn.Sequential(
            nn.Embedding(vocab_size, hidden_dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hidden_dim, nhead=8, batch_first=True),
                num_layers=2
            )
        )
        
        # Decoder: Z -> transition probabilities
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        
    def forward(self, x_0, t):
        """
        Compute information bottleneck guided transitions.
        
        Args:
            x_0: Original tokens (batch_size, seq_len)
            t: Current time
            
        Returns:
            transition_logits: Logits for transition probabilities
            info_loss: Information bottleneck loss
        """
        # Encode to compressed representation
        z = self.encoder(x_0)  # (batch_size, seq_len, hidden_dim)
        
        # Add noise based on time
        noise_scale = t if isinstance(t, float) else t.mean().item()
        z_noisy = z + torch.randn_like(z) * noise_scale
        
        # Decode to transition probabilities
        transition_logits = self.decoder(z_noisy)  # (batch_size, seq_len, vocab_size)
        
        # Compute information bottleneck loss
        # I(X_t; Y) - approximated by reconstruction ability
        reconstruction_loss = F.cross_entropy(
            transition_logits.reshape(-1, self.vocab_size),
            x_0.reshape(-1),
            reduction='mean'
        )
        
        # I(X_t; X_0) - approximated by KL divergence of latent
        kl_loss = -0.5 * torch.mean(
            1 + torch.log(z.var(dim=0) + 1e-8) - z.mean(dim=0).pow(2) - z.var(dim=0)
        )
        
        info_loss = -reconstruction_loss + self.beta * kl_loss
        
        return transition_logits, info_loss


class EnhancedAdaptiveUniformV2(AdaptiveUniform):
    """
    Enhanced AEGUD V2 with all fixes and new research directions.
    Implements:
    1. Fixed shape mismatches
    2. Vocabulary-aware decay
    3. Learnable convergence schedules
    4. Information bottleneck
    5. Relaxed convergence criteria
    """
    
    def __init__(self, dim, entropy_estimator=None, transition_matrix=None,
                 entropy_scale=1.0, sparsity_k=None,
                 # Original enhancement parameters
                 use_asymptotic_guarantee=True,
                 use_two_stage=True,
                 stage_transition_point=0.8,
                 use_controlled_decay=True,
                 decay_tau=0.1,
                 kl_regularization_weight=0.01,
                 # New research parameters
                 use_vocabulary_aware_decay=False,
                 use_learnable_schedule=False,
                 use_information_bottleneck=False,
                 info_bottleneck_beta=0.1,
                 relaxed_convergence_epsilon=0.1):
        """
        Args:
            All previous args plus:
            use_vocabulary_aware_decay: Enable vocabulary-aware decay rates
            use_learnable_schedule: Enable learnable convergence schedule
            use_information_bottleneck: Enable information bottleneck approach
            info_bottleneck_beta: Beta parameter for information bottleneck
            relaxed_convergence_epsilon: Epsilon for relaxed convergence criteria
        """
        super().__init__(dim, entropy_estimator, transition_matrix, entropy_scale, sparsity_k)
        
        # Original enhancement flags
        self.use_asymptotic_guarantee = use_asymptotic_guarantee
        self.use_two_stage = use_two_stage
        self.stage_transition_point = stage_transition_point
        self.use_controlled_decay = use_controlled_decay
        self.decay_tau = decay_tau
        self.kl_regularization_weight = kl_regularization_weight
        
        # New research components
        self.use_vocabulary_aware_decay = use_vocabulary_aware_decay
        self.use_learnable_schedule = use_learnable_schedule
        self.use_information_bottleneck = use_information_bottleneck
        self.relaxed_convergence_epsilon = relaxed_convergence_epsilon
        
        # Initialize new modules
        if use_vocabulary_aware_decay:
            self.vocab_decay = VocabularyAwareDecay(dim)
            
        if use_learnable_schedule:
            self.learnable_schedule = LearnableConvergenceSchedule()
            
        if use_information_bottleneck:
            self.info_bottleneck = InformationBottleneck(dim, beta=info_bottleneck_beta)
        
        # For tracking convergence metrics
        self.register_buffer('uniform_distribution', torch.ones(dim) / dim)
        
    def get_adaptive_weight(self, t, max_t=1.0, x_t=None):
        """
        Enhanced adaptive weight computation with new research directions.
        """
        # Use learnable schedule if enabled
        if self.use_learnable_schedule:
            return self.learnable_schedule(t, max_t)
        
        # Otherwise use original logic
        if not self.use_asymptotic_guarantee and not self.use_controlled_decay:
            return 1.0
        
        # Normalize t to [0, 1] range
        normalized_t = t / max_t if isinstance(t, (int, float)) else t
        
        # Convert to scalar if tensor
        if isinstance(normalized_t, torch.Tensor):
            normalized_t = normalized_t.item()
        
        if self.use_two_stage:
            # Two-stage approach
            if normalized_t < self.stage_transition_point:
                if self.use_controlled_decay:
                    decay_factor = np.exp(-normalized_t / (self.decay_tau * self.stage_transition_point))
                    return decay_factor
                else:
                    return 1.0
            else:
                return 0.0
        
        elif self.use_asymptotic_guarantee:
            # Smooth asymptotic guarantee
            uniform_weight = normalized_t ** 2
            adaptive_weight = 1 - uniform_weight
            
            if self.use_controlled_decay:
                decay_factor = np.exp(-normalized_t / self.decay_tau)
                adaptive_weight = adaptive_weight * decay_factor
            
            return adaptive_weight
        
        elif self.use_controlled_decay:
            return np.exp(-normalized_t / self.decay_tau)
        
        return 1.0
    
    def adaptive_rate(self, x_t, t, max_t=1.0):
        """
        Enhanced adaptive rate with vocabulary-aware decay and information bottleneck.
        """
        batch_size, seq_len = x_t.shape
        device = x_t.device
        
        # Get base adaptive weight
        adaptive_weight = self.get_adaptive_weight(t, max_t, x_t)
        
        # Apply vocabulary-aware decay if enabled
        if self.use_vocabulary_aware_decay and adaptive_weight > 0:
            vocab_weights = self.vocab_decay(x_t, t, self.decay_tau)
            # Modulate adaptive weight by vocabulary-specific weights
            adaptive_weight = adaptive_weight * vocab_weights.mean()
        
        # Use information bottleneck if enabled
        if self.use_information_bottleneck and adaptive_weight > 0:
            ib_logits, ib_loss = self.info_bottleneck(x_t, t)
            # Store IB loss for potential use in training
            self.last_ib_loss = ib_loss
        
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
    
    def score_entropy_with_kl(self, score, sigma, x_t, x_0, t=None, max_t=1.0):
        """
        Fixed score entropy with KL regularization.
        """
        # Get base score entropy loss - use parent's score_entropy method correctly
        base_loss = super().score_entropy(score, sigma, x_t, x_0)
        
        if self.kl_regularization_weight > 0 and t is not None:
            # Compute transition probabilities for KL regularization
            rate_matrix = self.adaptive_rate(x_t, t, max_t)
            
            # Convert rates to probabilities
            trans_probs = F.softmax(rate_matrix, dim=-1)
            
            # Compute KL divergence
            kl_loss = self.compute_kl_from_uniform(trans_probs)
            
            # Weight KL loss based on time
            time_weight = (t / max_t) ** 2 if isinstance(t, (int, float)) else (t ** 2).mean()
            kl_loss = kl_loss * time_weight * self.kl_regularization_weight
            
            # Add information bottleneck loss if available
            if hasattr(self, 'last_ib_loss'):
                kl_loss = kl_loss + 0.1 * self.last_ib_loss
            
            # Add to base loss
            total_loss = base_loss + kl_loss
            
            # Store metrics for logging
            self.last_kl_loss = kl_loss.item() if hasattr(kl_loss, 'item') else kl_loss
            
            return total_loss
        
        return base_loss
    
    def compute_kl_from_uniform(self, transition_probs):
        """
        Compute KL divergence from uniform distribution.
        """
        batch_size, seq_len, vocab_size, _ = transition_probs.shape
        
        # Compute marginal by averaging transitions
        marginal = transition_probs.mean(dim=2)  # (B, L, V)
        
        # Compute KL divergence from uniform
        uniform = self.uniform_distribution.unsqueeze(0).unsqueeze(0)
        
        # KL(P||Q) = sum(P * log(P/Q))
        kl = F.kl_div(
            uniform.log(),
            marginal,
            reduction='batchmean'
        )
        
        return kl
    
    def check_relaxed_convergence(self, distribution):
        """
        Check convergence with relaxed criteria.
        Accept distributions that are "close enough" to uniform.
        """
        uniform = 1.0 / self._dim
        max_deviation = torch.max(torch.abs(distribution - uniform))
        
        # Check if within epsilon of uniform
        converged = max_deviation < self.relaxed_convergence_epsilon * uniform
        
        return converged, max_deviation.item()
    
    def get_convergence_metrics(self, x_t, t, max_t=1.0):
        """
        Enhanced convergence metrics with relaxed criteria.
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
        ) / math.log(self._dim)
        
        # Compute KL from uniform
        uniform = torch.ones_like(token_probs) / self._dim
        kl_from_uniform = F.kl_div(
            uniform.log(),
            token_probs,
            reduction='batchmean'
        )
        
        # Check relaxed convergence
        avg_probs = token_probs.mean(dim=0)
        relaxed_converged, max_dev = self.check_relaxed_convergence(avg_probs)
        
        # Estimate mutual information
        if hasattr(self, 'entropy_estimator'):
            info_content = self.entropy_estimator.estimate_information_content(x_t)
            mutual_information = info_content * (1 - t / max_t)
        else:
            mutual_information = torch.tensor(0.0)
        
        # Get effective temperature
        adaptive_weight = self.get_adaptive_weight(t, max_t, x_t)
        effective_temp = 1.0 / (adaptive_weight + 1e-6) if adaptive_weight > 0 else float('inf')
        
        return {
            'mutual_information': mutual_information.mean().item(),
            'kl_from_uniform': kl_from_uniform.item(),
            'entropy': entropy.mean().item(),
            'effective_temperature': effective_temp,
            'relaxed_converged': relaxed_converged,
            'max_deviation': max_dev
        }