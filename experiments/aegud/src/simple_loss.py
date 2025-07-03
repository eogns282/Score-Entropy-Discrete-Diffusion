"""
Simple loss wrapper for experiments
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from losses import get_loss_fn
from model import utils as mutils


def score_entropy_loss(model, x_0, graph, noise, sampling_eps=1e-3):
    """
    Simple wrapper around the loss function for experiments.
    
    Args:
        model: The score model
        x_0: Clean data batch
        graph: Graph instance
        noise: Noise schedule instance
        
    Returns:
        loss: Scalar loss value
    """
    device = x_0.device
    batch_size = x_0.shape[0]
    
    # Sample random time steps
    t = (1 - sampling_eps) * torch.rand(batch_size, device=device) + sampling_eps
    
    # Get noise values
    sigma, dsigma = noise(t)
    
    # Sample perturbed data
    perturbed_batch = graph.sample_transition(x_0, sigma[:, None])
    
    # Get score function
    log_score_fn = mutils.get_score_fn(model, train=True, sampling=False)
    log_score = log_score_fn(perturbed_batch, sigma)
    
    # Compute score entropy loss
    loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, x_0)
    
    # Weight by dsigma
    loss = (dsigma[:, None] * loss).sum(dim=-1)
    
    return loss.mean()


class SimpleLoss:
    """Simple loss class wrapper for experiments."""
    
    def __init__(self):
        pass
    
    def __call__(self, score, x_t, x_0, sigma, graph):
        """
        Compute loss given score and other inputs.
        
        Args:
            score: Model output (log scores)
            x_t: Perturbed data
            x_0: Clean data
            sigma: Noise level
            graph: Graph instance
            
        Returns:
            loss: Scalar loss
        """
        # Compute score entropy loss
        # sigma is already shape (batch_size,), so add dimension for broadcasting
        loss = graph.score_entropy(score, sigma[:, None], x_t, x_0)
        
        # Average over sequence length and batch
        return loss.mean()