"""
Hierarchical Diffusion for AEGUD
Implements multi-scale diffusion with different granularity levels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
import math

from experiments.aegud.src.enhanced_adaptive_uniform_graph_v2 import EnhancedAdaptiveUniformV2
from experiments.aegud.src.entropy_estimator import EntropyEstimator, AdaptiveTransitionMatrix


class HierarchicalDiffusion(nn.Module):
    """
    Hierarchical diffusion that operates at multiple scales:
    - Level 0: Topic/style tokens (coarse-grained)
    - Level 1: Syntactic structure tokens
    - Level 2: Word-level tokens (fine-grained)
    """
    
    def __init__(self, vocab_size, num_levels=3, compression_ratios=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_levels = num_levels
        
        if compression_ratios is None:
            # Default compression ratios for each level
            compression_ratios = [8, 4, 1]  # Level 0 compresses by 8x, level 1 by 4x
        self.compression_ratios = compression_ratios
        
        # Create hierarchical vocabularies
        self.level_vocab_sizes = []
        for level, ratio in enumerate(compression_ratios):
            level_vocab_size = max(16, vocab_size // ratio)
            self.level_vocab_sizes.append(level_vocab_size)
        
        # Create graphs for each level
        self.level_graphs = nn.ModuleList()
        for level, level_vocab_size in enumerate(self.level_vocab_sizes):
            # Use more aggressive diffusion for coarser levels
            entropy_scale = 1.0 + 0.5 * (self.num_levels - level - 1)
            
            graph = EnhancedAdaptiveUniformV2(
                dim=level_vocab_size,
                entropy_scale=entropy_scale,
                use_two_stage=True,
                stage_transition_point=0.7 + 0.1 * level,  # Earlier transition for coarser levels
                use_controlled_decay=True,
                decay_tau=0.1 * (level + 1),  # Slower decay for finer levels
                use_vocabulary_aware_decay=True,
                relaxed_convergence_epsilon=0.1 + 0.05 * level
            )
            self.level_graphs.append(graph)
        
        # Token mapping between levels
        self.level_encoders = nn.ModuleList()
        self.level_decoders = nn.ModuleList()
        
        for level in range(self.num_levels - 1):
            # Encoder: fine level -> coarse level
            encoder = nn.Sequential(
                nn.Embedding(self.level_vocab_sizes[level + 1], 256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, self.level_vocab_sizes[level])
            )
            self.level_encoders.append(encoder)
            
            # Decoder: coarse level -> fine level
            decoder = nn.Sequential(
                nn.Embedding(self.level_vocab_sizes[level], 256),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, self.level_vocab_sizes[level + 1])
            )
            self.level_decoders.append(decoder)
    
    def encode_to_level(self, tokens, target_level):
        """
        Encode tokens to a coarser level.
        
        Args:
            tokens: Input tokens at finest level
            target_level: Target hierarchical level (0 = coarsest)
            
        Returns:
            encoded_tokens: Tokens at target level
        """
        current_tokens = tokens
        current_level = self.num_levels - 1
        
        while current_level > target_level:
            encoder = self.level_encoders[current_level - 1]
            
            # Get embeddings and map to coarser vocabulary
            logits = encoder(current_tokens)
            current_tokens = torch.argmax(logits, dim=-1)
            
            current_level -= 1
        
        return current_tokens
    
    def decode_from_level(self, tokens, source_level):
        """
        Decode tokens from a coarser level to finest level.
        
        Args:
            tokens: Input tokens at source level
            source_level: Source hierarchical level
            
        Returns:
            decoded_tokens: Tokens at finest level
        """
        current_tokens = tokens
        current_level = source_level
        
        while current_level < self.num_levels - 1:
            decoder = self.level_decoders[current_level]
            
            # Get embeddings and map to finer vocabulary
            logits = decoder(current_tokens)
            current_tokens = torch.argmax(logits, dim=-1)
            
            current_level += 1
        
        return current_tokens
    
    def hierarchical_diffusion_step(self, x_0, t, noise_scales):
        """
        Perform hierarchical diffusion step.
        
        Args:
            x_0: Original tokens at finest level
            t: Time step
            noise_scales: Noise scales for each level
            
        Returns:
            x_t: Noised tokens at finest level
            level_outputs: Outputs at each hierarchical level
        """
        level_outputs = []
        
        # Process each level
        for level in range(self.num_levels):
            # Encode to current level
            if level < self.num_levels - 1:
                level_tokens = self.encode_to_level(x_0, level)
            else:
                level_tokens = x_0
            
            # Apply diffusion at this level
            graph = self.level_graphs[level]
            sigma = noise_scales[level]
            
            # Sample transition
            level_x_t = graph.sample_transition(level_tokens, sigma)
            
            # Store output
            level_outputs.append({
                'tokens': level_x_t,
                'level': level,
                'sigma': sigma
            })
        
        # Combine information from all levels
        # Start from coarsest level and refine
        combined_tokens = level_outputs[0]['tokens']
        
        for level in range(1, self.num_levels):
            # Decode coarse tokens
            decoded_coarse = self.decode_from_level(combined_tokens, level - 1)
            
            # Blend with current level based on noise
            blend_weight = torch.exp(-noise_scales[level])
            
            if level < self.num_levels - 1:
                current_tokens = self.encode_to_level(level_outputs[level]['tokens'], level - 1)
            else:
                current_tokens = level_outputs[level]['tokens']
            
            # Stochastic blending
            use_coarse = torch.rand_like(current_tokens, dtype=torch.float) < (1 - blend_weight)
            combined_tokens = torch.where(use_coarse, decoded_coarse, current_tokens)
        
        return combined_tokens, level_outputs
    
    def get_hierarchical_noise_schedule(self, t, max_t=1.0):
        """
        Get noise scales for each hierarchical level.
        Coarser levels get more noise earlier.
        """
        normalized_t = t / max_t if isinstance(t, (int, float)) else t
        
        noise_scales = []
        for level in range(self.num_levels):
            # Coarser levels transition faster
            level_speed = 1.0 + 0.5 * (self.num_levels - level - 1)
            level_t = min(1.0, normalized_t * level_speed)
            
            # Different noise schedules for different levels
            if level == 0:
                # Coarse level: rapid transition
                sigma = -np.log(1 - level_t * 0.99)
            elif level == 1:
                # Middle level: moderate transition
                sigma = -np.log(1 - level_t * 0.9)
            else:
                # Fine level: slow transition
                sigma = -np.log(1 - level_t * 0.8)
            
            noise_scales.append(torch.tensor(sigma))
        
        return noise_scales
    
    def compute_hierarchical_loss(self, model, x_0, t):
        """
        Compute loss across all hierarchical levels.
        """
        # Get noise scales for each level
        noise_scales = self.get_hierarchical_noise_schedule(t)
        
        # Perform hierarchical diffusion
        x_t, level_outputs = self.hierarchical_diffusion_step(x_0, t, noise_scales)
        
        total_loss = 0.0
        level_weights = [1.0, 0.5, 0.25]  # Decreasing importance for finer levels
        
        for level, output in enumerate(level_outputs):
            if level < self.num_levels - 1:
                # For coarse levels, use encoded model predictions
                level_x_0 = self.encode_to_level(x_0, level)
            else:
                level_x_0 = x_0
            
            # Get graph and compute score entropy
            graph = self.level_graphs[level]
            level_tokens = output['tokens']
            sigma = output['sigma'].unsqueeze(0)
            
            # Model prediction at this level (simplified - in practice would need level-specific model)
            # Here we assume the model can handle different vocabulary sizes
            score = model(level_tokens, sigma)
            
            # Compute loss
            if hasattr(graph, 'score_entropy_with_kl'):
                loss = graph.score_entropy_with_kl(
                    score, sigma, level_tokens, level_x_0, 
                    t=t, max_t=1.0
                )
            else:
                loss = graph.score_entropy(score, sigma, level_tokens, level_x_0)
            
            total_loss += level_weights[level] * loss.mean()
        
        return total_loss
    
    def validate_hierarchical_convergence(self, num_steps=50, device='cuda'):
        """
        Validate convergence at each hierarchical level.
        """
        print("Validating hierarchical convergence...")
        
        results = {}
        
        for level in range(self.num_levels):
            graph = self.level_graphs[level]
            vocab_size = self.level_vocab_sizes[level]
            
            print(f"\nLevel {level} (vocab_size={vocab_size}):")
            
            # Test convergence at this level
            metrics_over_time = graph.validate_convergence(
                num_steps=num_steps,
                vocab_size=vocab_size,
                seq_len=32 // (self.compression_ratios[level] or 1),
                device=device
            )
            
            results[f'level_{level}'] = metrics_over_time[0]
            
            # Check final convergence
            final_kl = metrics_over_time[0]['kl_from_uniform'][-1]
            final_entropy = metrics_over_time[0]['entropy'][-1]
            converged = metrics_over_time[1]
            
            print(f"  Final KL: {final_kl:.6f}")
            print(f"  Final entropy: {final_entropy:.6f}")
            print(f"  Converged: {converged}")
        
        return results


class SemanticAwareTransitions(nn.Module):
    """
    Semantic-aware transition matrix that preserves meaning during diffusion.
    """
    
    def __init__(self, vocab_size, embedding_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Semantic embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Context encoder
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, nhead=8, batch_first=True),
            num_layers=2
        )
        
        # Semantic similarity temperature
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def compute_semantic_transitions(self, x_t, mask=None):
        """
        Compute transition probabilities based on semantic similarity.
        """
        # Get token embeddings
        embeddings = self.token_embeddings(x_t)  # (batch, seq_len, embed_dim)
        
        # Encode context
        if mask is not None:
            context = self.context_encoder(embeddings, src_key_padding_mask=mask)
        else:
            context = self.context_encoder(embeddings)
        
        # Compute semantic similarity matrix
        all_embeddings = self.token_embeddings.weight  # (vocab_size, embed_dim)
        
        # For each position, compute similarity to all vocabulary tokens
        # context: (batch, seq_len, embed_dim)
        # all_embeddings: (vocab_size, embed_dim)
        
        similarity = torch.matmul(context, all_embeddings.T) / self.temperature
        # similarity: (batch, seq_len, vocab_size)
        
        # Convert to transition probabilities
        transition_probs = F.softmax(similarity, dim=-1)
        
        # Create full transition matrix
        batch_size, seq_len = x_t.shape
        trans_matrix = torch.zeros(batch_size, seq_len, self.vocab_size, self.vocab_size, 
                                  device=x_t.device)
        
        # Fill transition matrix based on semantic similarities
        for i in range(self.vocab_size):
            trans_matrix[:, :, i, :] = transition_probs
        
        return trans_matrix