"""
Adaptive Loss Functions for AEGUD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from losses import get_loss_fn
from model import utils as mutils


def adaptive_score_entropy_loss(model, x_0, graph, noise, 
                               entropy_regularizer=0.01,
                               info_preservation_weight=0.1,
                               transition_smoothness_weight=0.001,
                               sampling_eps=1e-3):
    """
    Enhanced score entropy loss with adaptive components.
    
    Args:
        model: The diffusion model
        x_0: Original data (batch_size, seq_len)
        graph: Adaptive graph instance
        noise: Noise schedule instance
        entropy_regularizer: Weight for entropy regularization
        info_preservation_weight: Weight for information preservation loss
        transition_smoothness_weight: Weight for transition smoothness
        
    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary of individual loss components
    """
    device = x_0.device
    batch_size, seq_len = x_0.shape
    
    # Base score entropy loss - compute it directly
    t = (1 - sampling_eps) * torch.rand(batch_size, device=device) + sampling_eps
    sigma, dsigma = noise(t)
    perturbed_batch = graph.sample_transition(x_0, sigma[:, None])
    
    log_score_fn = mutils.get_score_fn(model, train=True, sampling=False)
    log_score = log_score_fn(perturbed_batch, sigma)
    
    loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, x_0)
    base_loss = (dsigma[:, None] * loss).sum(dim=-1).mean()
    
    # For now, skip the complex adaptive sampling in forward diffusion
    x_t = perturbed_batch
        
    # Information preservation loss
    if hasattr(graph, 'entropy_estimator'):
        # Original information content
        info_x0 = graph.entropy_estimator.estimate_information_content(x_0)
        
        # Information after diffusion
        info_xt = graph.entropy_estimator.estimate_information_content(x_t)
        
        # Preserve information: penalize large drops in information
        info_loss = F.relu(info_x0 - info_xt).mean()
    else:
        info_loss = torch.tensor(0.0, device=device)
        
    # Entropy regularization
    if hasattr(graph, 'entropy_estimator'):
        # Get entropy predictions
        entropy_scores = graph.entropy_estimator(x_t, return_all=True)
        
        # Regularize entropy to be diverse but not extreme
        target_entropy = 0.5  # Target moderate entropy
        entropy_reg = ((entropy_scores - target_entropy) ** 2).mean()
    else:
        entropy_reg = torch.tensor(0.0, device=device)
        
    # Transition smoothness loss - simplified for memory efficiency
    if hasattr(graph, 'transition_matrix'):
        # Just use a simple regularization on the embeddings
        embeddings = graph.transition_matrix.token_embeddings.weight
        smoothness_loss = embeddings.norm(p=2, dim=1).mean() * 0.01
    else:
        smoothness_loss = torch.tensor(0.0, device=device)
        
    # Combine losses
    total_loss = (base_loss + 
                  entropy_regularizer * entropy_reg +
                  info_preservation_weight * info_loss +
                  transition_smoothness_weight * smoothness_loss)
    
    # Return detailed loss breakdown
    loss_dict = {
        'total': total_loss.item(),
        'base_score_entropy': base_loss.item(),
        'entropy_regularization': entropy_reg.item(),
        'info_preservation': info_loss.item(),
        'transition_smoothness': smoothness_loss.item()
    }
    
    return total_loss, loss_dict


def hierarchical_loss(model, x_0, graph, noise, level_weights=None):
    """
    Loss function for hierarchical adaptive uniform diffusion.
    
    Args:
        model: The diffusion model
        x_0: Original data
        graph: HierarchicalAdaptiveUniform instance
        noise: Noise schedule
        level_weights: Optional weights for different hierarchy levels
        
    Returns:
        total_loss: Combined hierarchical loss
        loss_dict: Loss components
    """
    if level_weights is None:
        level_weights = [1.0, 0.5, 0.25]  # Decreasing weights for finer levels
        
    device = x_0.device
    batch_size = x_0.shape[0]
    
    # Compute losses at different levels
    level_losses = []
    
    for level in range(min(len(level_weights), 3)):
        # Different time scales for different levels
        if level == 0:
            # Coarse level - focus on early times (large noise)
            t = torch.rand(batch_size, device=device) * 0.3 + 0.7
        elif level == 1:
            # Middle level - middle times
            t = torch.rand(batch_size, device=device) * 0.4 + 0.3
        else:
            # Fine level - late times (small noise)
            t = torch.rand(batch_size, device=device) * 0.3
            
        # Compute loss for this level
        sigma, dsigma = noise(t)
        perturbed_batch = graph.sample_transition(x_0, sigma[:, None])
        
        log_score_fn = mutils.get_score_fn(model, train=True, sampling=False)
        log_score = log_score_fn(perturbed_batch, sigma)
        
        loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, x_0)
        level_loss = (dsigma[:, None] * loss).sum(dim=-1).mean()
        
        level_losses.append(level_loss * level_weights[level])
        
    # Combine level losses
    total_loss = sum(level_losses)
    
    loss_dict = {
        'total': total_loss.item(),
        'level_0': level_losses[0].item() if len(level_losses) > 0 else 0,
        'level_1': level_losses[1].item() if len(level_losses) > 1 else 0,
        'level_2': level_losses[2].item() if len(level_losses) > 2 else 0,
    }
    
    return total_loss, loss_dict


class InfoNCEAuxiliaryLoss(nn.Module):
    """
    Contrastive loss to ensure the model learns meaningful representations.
    """
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, model, x_0, graph, noise):
        """
        Compute InfoNCE loss between clean and noised representations.
        """
        device = x_0.device
        batch_size = x_0.shape[0]
        
        # Get two different noise levels
        t1 = torch.rand(batch_size, device=device) * 0.3  # Low noise
        t2 = torch.rand(batch_size, device=device) * 0.3 + 0.5  # High noise
        
        # Get representations at different noise levels
        # This assumes the model has an encode method or we can extract features
        with torch.no_grad():
            # Get noised versions
            sigma1 = noise.sigma(t1)
            sigma2 = noise.sigma(t2)
            
            # For now, we'll use the model's output as representation
            # In practice, you might want to add an encoder head
            score1 = model(x_0, sigma1)
            score2 = model(x_0, sigma2)
            
        # Global pooling to get sequence representations
        repr1 = score1.mean(dim=1)  # (B, D)
        repr2 = score2.mean(dim=1)  # (B, D)
        
        # Normalize representations
        repr1 = F.normalize(repr1, dim=-1)
        repr2 = F.normalize(repr2, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(repr1, repr2.T) / self.temperature
        
        # InfoNCE loss
        labels = torch.arange(batch_size, device=device)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


def diversity_promoting_loss(model, x_0, graph, noise, diversity_weight=0.01):
    """
    Encourage diverse transitions in the adaptive uniform model.
    """
    device = x_0.device
    batch_size = x_0.shape[0]
    
    if not hasattr(graph, 'transition_matrix'):
        return torch.tensor(0.0, device=device)
        
    # Get transition matrix
    trans_matrix = graph.transition_matrix()  # (V, V)
    
    # Compute entropy of each row (transition distribution for each token)
    trans_entropy = -torch.sum(
        trans_matrix * torch.log(trans_matrix + 1e-10), 
        dim=-1
    )
    
    # Encourage high entropy (diverse transitions)
    max_entropy = torch.log(torch.tensor(trans_matrix.shape[0], dtype=torch.float))
    diversity_loss = (max_entropy - trans_entropy).mean()
    
    return diversity_weight * diversity_loss