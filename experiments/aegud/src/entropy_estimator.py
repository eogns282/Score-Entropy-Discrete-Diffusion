"""
Entropy Estimation Module for Adaptive Uniform Diffusion
This module estimates local entropy to guide adaptive transitions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class EntropyEstimator(nn.Module):
    """Estimates local entropy for each position in a sequence."""
    
    def __init__(self, vocab_size, hidden_dim=256, num_layers=2, num_heads=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, 1024, hidden_dim) * 0.02)
        
        # Context encoder using transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Entropy prediction heads
        self.local_entropy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Global context aggregator
        self.global_context = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
    def forward(self, x_t, return_all=False):
        """
        Estimate entropy for each position in the sequence.
        
        Args:
            x_t: Token indices of shape (batch_size, seq_len)
            return_all: If True, return entropy for all positions
            
        Returns:
            entropy: Shape (batch_size, seq_len) if return_all else (batch_size,)
        """
        batch_size, seq_len = x_t.shape
        
        # Embed tokens
        x_emb = self.token_embedding(x_t)  # (B, L, D)
        
        # Add position embeddings
        pos_emb = self.position_embedding[:, :seq_len, :]
        x_emb = x_emb + pos_emb
        
        # Encode context
        context = self.context_encoder(x_emb)  # (B, L, D)
        
        # Estimate local entropy
        local_entropy = self.local_entropy_head(context)  # (B, L, 1)
        local_entropy = local_entropy.squeeze(-1)  # (B, L)
        
        if return_all:
            return local_entropy
        else:
            # Return average entropy across sequence
            return local_entropy.mean(dim=1)
    
    def estimate_information_content(self, x_t):
        """
        Estimate information content based on token diversity and patterns.
        
        Args:
            x_t: Token indices of shape (batch_size, seq_len)
            
        Returns:
            info_content: Scalar between 0 and 1
        """
        batch_size, seq_len = x_t.shape
        
        # Calculate token frequency entropy
        token_counts = torch.zeros(batch_size, self.vocab_size, device=x_t.device)
        token_counts.scatter_add_(1, x_t, torch.ones_like(x_t, dtype=torch.float))
        
        # Normalize to get probabilities
        token_probs = token_counts / seq_len
        
        # Calculate entropy (avoid log(0))
        token_entropy = -torch.sum(
            token_probs * torch.log(token_probs + 1e-10), 
            dim=1
        ) / math.log(self.vocab_size)  # Normalize to [0, 1]
        
        # Get contextual entropy
        local_entropy = self.forward(x_t, return_all=False)
        
        # Combine frequency and contextual entropy
        info_content = 0.5 * token_entropy + 0.5 * local_entropy
        
        return info_content


class AdaptiveTransitionMatrix(nn.Module):
    """Learns adaptive transition probabilities based on context."""
    
    def __init__(self, vocab_size, hidden_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Token embeddings for semantic similarity
        self.token_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.token_embeddings.weight.data.normal_(0, 0.02)
        
        # Temperature parameter for controlling sharpness
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Learnable bias for self-transitions
        self.self_transition_bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, entropy_scores=None):
        """
        Compute adaptive transition matrix.
        
        Args:
            entropy_scores: Optional entropy scores to modulate transitions
            
        Returns:
            transition_matrix: Shape (vocab_size, vocab_size)
        """
        # Compute semantic similarity matrix
        embeddings = self.token_embeddings.weight  # (V, D)
        similarity = torch.matmul(embeddings, embeddings.T)  # (V, V)
        
        # Normalize by embedding dimension
        similarity = similarity / math.sqrt(embeddings.shape[1])
        
        # Add self-transition bias
        similarity = similarity + torch.eye(
            self.vocab_size, device=similarity.device
        ) * self.self_transition_bias
        
        # Apply temperature scaling
        if entropy_scores is not None:
            # Higher entropy -> higher temperature -> more uniform
            temp = self.temperature * (1.0 + entropy_scores.mean())
        else:
            temp = self.temperature
            
        # Convert to probabilities
        transition_probs = F.softmax(similarity / temp, dim=-1)
        
        return transition_probs
    
    def get_sparse_transitions(self, k=10):
        """
        Get sparse transition matrix keeping only top-k transitions per token.
        
        Args:
            k: Number of top transitions to keep
            
        Returns:
            sparse_transitions: Sparse transition matrix
        """
        with torch.no_grad():
            transition_matrix = self.forward()
            
            # Keep only top-k values per row
            topk_vals, topk_idx = torch.topk(transition_matrix, k, dim=-1)
            
            # Create sparse matrix
            sparse_transitions = torch.zeros_like(transition_matrix)
            sparse_transitions.scatter_(1, topk_idx, topk_vals)
            
            # Renormalize
            sparse_transitions = sparse_transitions / sparse_transitions.sum(dim=-1, keepdim=True)
            
        return sparse_transitions