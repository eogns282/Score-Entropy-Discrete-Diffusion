"""
Adaptive Uniform Graph for AEGUD (Adaptive Entropy-Guided Uniform Diffusion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import sys
import os

# Add parent directory to path to import from main codebase
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from graph_lib import Graph, Uniform
from experiments.aegud.src.entropy_estimator import EntropyEstimator, AdaptiveTransitionMatrix


class AdaptiveUniform(Graph, nn.Module):
    """
    Adaptive Uniform Graph that modulates transitions based on entropy.
    """
    
    def __init__(self, dim, entropy_estimator=None, transition_matrix=None, 
                 entropy_scale=1.0, sparsity_k=None):
        """
        Args:
            dim: Vocabulary size
            entropy_estimator: EntropyEstimator module
            transition_matrix: AdaptiveTransitionMatrix module
            entropy_scale: Scale factor for entropy influence
            sparsity_k: If set, use sparse transitions with k neighbors
        """
        Graph.__init__(self)
        nn.Module.__init__(self)
        self._dim = dim
        self.entropy_scale = entropy_scale
        self.sparsity_k = sparsity_k
        
        # Initialize entropy estimator if not provided
        if entropy_estimator is None:
            self.entropy_estimator = EntropyEstimator(dim)
        else:
            self.entropy_estimator = entropy_estimator
            
        # Initialize transition matrix if not provided
        if transition_matrix is None:
            self.transition_matrix = AdaptiveTransitionMatrix(dim)
        else:
            self.transition_matrix = transition_matrix
            
        # Base uniform rate (will be modulated)
        self.base_rate = 1.0 / dim
        
        # Learnable parameters for rate modulation
        self.rate_scale = nn.Parameter(torch.ones(1))
        self.rate_bias = nn.Parameter(torch.zeros(1))
        
        # Register as nn.Module to handle device placement
        self.register_buffer('_dim_tensor', torch.tensor(self._dim))
        
    @property
    def dim(self):
        return self._dim
    
    def rate(self, i, j):
        """Base rate function (will be overridden by adaptive version)."""
        if i == j:
            return -1.0
        else:
            return self.base_rate * self.rate_scale + self.rate_bias
    
    def adaptive_rate(self, x_t, t):
        """
        Compute adaptive rate matrix based on current state and time.
        
        Args:
            x_t: Current token indices (batch_size, seq_len)
            t: Current time step
            
        Returns:
            rate_matrix: Shape (batch_size, seq_len, vocab_size, vocab_size)
        """
        batch_size, seq_len = x_t.shape
        
        # Get entropy scores for each position
        entropy_scores = self.entropy_estimator(x_t, return_all=True)  # (B, L)
        
        # Get adaptive transition matrix
        if self.sparsity_k is not None:
            trans_matrix = self.transition_matrix.get_sparse_transitions(self.sparsity_k)
        else:
            trans_matrix = self.transition_matrix(entropy_scores)  # (V, V)
        
        # Expand transition matrix for batch
        trans_matrix = trans_matrix.unsqueeze(0).unsqueeze(0)  # (1, 1, V, V)
        trans_matrix = trans_matrix.expand(batch_size, seq_len, -1, -1)  # (B, L, V, V)
        
        # Modulate by entropy: high entropy → more uniform transitions
        entropy_factor = 1.0 + self.entropy_scale * entropy_scores  # (B, L)
        entropy_factor = entropy_factor.unsqueeze(-1).unsqueeze(-1)  # (B, L, 1, 1)
        
        # Compute rate matrix
        rate_matrix = trans_matrix * self.rate_scale * entropy_factor
        
        # Set diagonal to negative sum of off-diagonal
        mask = torch.eye(self._dim, device=rate_matrix.device).bool()
        mask = mask.unsqueeze(0).unsqueeze(0).expand_as(rate_matrix)
        
        off_diagonal_sum = rate_matrix.masked_fill(mask, 0).sum(dim=-1, keepdim=True)
        rate_matrix = rate_matrix.masked_fill(mask, 0) - off_diagonal_sum * mask.float()
        
        return rate_matrix
    
    def transp_rate(self, i, j):
        """Transpose rate (symmetric for uniform)."""
        return self.rate(i, j)
    
    def sample_transition(self, i, sigma):
        """Sample transitions - use base uniform implementation for now."""
        # Use the simple uniform sampling for PoC
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, torch.randint_like(i, self.dim), i)
        return i_pert
    
    def sample_transition_adaptive(self, x_t, positions, sigma, t):
        """
        Sample transitions adaptively based on context.
        
        Args:
            x_t: Current tokens (batch_size, seq_len)
            positions: Positions to sample for (batch_size,) 
            sigma: Noise level
            t: Time step
            
        Returns:
            new_tokens: Sampled tokens for specified positions
        """
        batch_size = x_t.shape[0]
        
        # Get adaptive rate matrix
        rate_matrix = self.adaptive_rate(x_t, t)  # (B, L, V, V)
        
        # Extract rates for specified positions
        position_rates = []
        for b in range(batch_size):
            pos = positions[b]
            current_token = x_t[b, pos]
            rates = rate_matrix[b, pos, current_token, :]  # (V,)
            position_rates.append(rates)
            
        position_rates = torch.stack(position_rates)  # (B, V)
        
        # Convert to probabilities
        trans_probs = F.softmax(position_rates * sigma, dim=-1)
        
        # Sample new tokens
        new_tokens = torch.multinomial(trans_probs, 1).squeeze(-1)
        
        return new_tokens
    
    def staggered_score(self, score, dsigma_dt, sigma, x_t, t):
        """
        Compute staggered score with adaptive adjustments.
        
        Args:
            score: Base score
            dsigma_dt: Derivative of sigma
            sigma: Current sigma
            x_t: Current tokens
            t: Time step
            
        Returns:
            Adjusted staggered score
        """
        # Get entropy-based adjustment
        entropy_scores = self.entropy_estimator(x_t, return_all=False)  # (B,)
        
        # High entropy regions need less adjustment
        adjustment_factor = 1.0 - self.entropy_scale * entropy_scores.mean()
        
        # Apply adjustment to staggered score computation
        return score * adjustment_factor
    
    def transition(self, i, sigma):
        """
        Computes the transition matrix - compatible with batch inputs.
        """
        # Use the base uniform transition logic for batches
        device = i.device if isinstance(i, torch.Tensor) else torch.device('cpu')
        
        trans = torch.ones(*i.shape, self.dim, device=device) * (1 - (-sigma[..., None]).exp()) / self.dim
        trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans))
        trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
        return trans
    
    def sample_limit(self, *batch_dims):
        """
        Sample the limiting distribution (uniform for this graph).
        """
        limit_dist = torch.ones(*batch_dims, self._dim) / self._dim
        samples = torch.multinomial(limit_dist.view(-1, self._dim), 1).view(*batch_dims)
        return samples, limit_dist
    
    def score_entropy(self, score, sigma, x_t, x_0):
        """
        Compute score entropy with adaptive modifications.
        
        This is where AEGUD's key innovation happens - we modify the
        score entropy calculation based on local information content.
        """
        # Base uniform score entropy calculation
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        
        ratio = score.exp().clamp(0, 100)
        ratio_all = ratio.sum(dim=-1, keepdim=True)
        
        loss = -torch.log(ratio.gather(-1, x_0[..., None]) / ratio_all)
        loss = loss.squeeze(-1)  # Remove the last dimension to get (B, L)
        
        # Adaptive modifications
        if hasattr(self, 'entropy_estimator') and self.entropy_estimator is not None:
            # Get entropy scores
            entropy_scores = self.entropy_estimator(x_t, return_all=True)  # (B, L)
            
            # Get information content
            info_content = self.entropy_estimator.estimate_information_content(x_t)  # (B,)
            
            # Adaptive modification based on information content
            info_factor = 0.5 + info_content.unsqueeze(-1)  # (B, 1)
            
            # Position-wise adjustment based on local entropy
            position_factor = 0.5 + entropy_scores.mean(dim=1, keepdim=True)  # (B, 1)
            
            # Apply adjustments - loss shape should be (B, L)
            loss = loss * info_factor * position_factor
            
        return loss / esigm1


class HierarchicalAdaptiveUniform(AdaptiveUniform):
    """
    Hierarchical version of Adaptive Uniform that operates at multiple scales.
    """
    
    def __init__(self, dim, num_levels=3, **kwargs):
        super().__init__(dim, **kwargs)
        self.num_levels = num_levels
        
        # Create hierarchical entropy estimators
        self.level_estimators = nn.ModuleList([
            EntropyEstimator(dim, hidden_dim=256 * (2 ** i))
            for i in range(num_levels)
        ])
        
        # Level mixing weights
        self.level_weights = nn.Parameter(torch.ones(num_levels) / num_levels)
        
    def hierarchical_entropy(self, x_t):
        """
        Compute multi-scale entropy estimates.
        
        Args:
            x_t: Token indices (batch_size, seq_len)
            
        Returns:
            combined_entropy: Weighted combination of entropy at different scales
        """
        entropies = []
        
        for i, estimator in enumerate(self.level_estimators):
            level_entropy = estimator(x_t, return_all=True)
            entropies.append(level_entropy)
            
        # Stack and weight
        entropies = torch.stack(entropies, dim=0)  # (num_levels, B, L)
        weights = F.softmax(self.level_weights, dim=0).view(-1, 1, 1)
        
        combined_entropy = (entropies * weights).sum(dim=0)  # (B, L)
        
        return combined_entropy
    
    def adaptive_rate(self, x_t, t):
        """Override to use hierarchical entropy."""
        # Use hierarchical entropy instead of single-level
        entropy_scores = self.hierarchical_entropy(x_t)
        
        # Rest follows parent implementation
        batch_size, seq_len = x_t.shape
        
        if self.sparsity_k is not None:
            trans_matrix = self.transition_matrix.get_sparse_transitions(self.sparsity_k)
        else:
            trans_matrix = self.transition_matrix(entropy_scores)
            
        trans_matrix = trans_matrix.unsqueeze(0).unsqueeze(0)
        trans_matrix = trans_matrix.expand(batch_size, seq_len, -1, -1)
        
        entropy_factor = 1.0 + self.entropy_scale * entropy_scores
        entropy_factor = entropy_factor.unsqueeze(-1).unsqueeze(-1)
        
        rate_matrix = trans_matrix * self.rate_scale * entropy_factor
        
        mask = torch.eye(self._dim, device=rate_matrix.device).bool()
        mask = mask.unsqueeze(0).unsqueeze(0).expand_as(rate_matrix)
        
        off_diagonal_sum = rate_matrix.masked_fill(mask, 0).sum(dim=-1, keepdim=True)
        rate_matrix = rate_matrix.masked_fill(mask, 0) - off_diagonal_sum * mask.float()
        
        return rate_matrix