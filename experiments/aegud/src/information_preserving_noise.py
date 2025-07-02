"""
Information-Preserving Noise Schedule for AEGUD
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from noise_lib import Noise, GeometricNoise, LogLinearNoise


class InformationPreservingNoise(Noise):
    """
    Adaptive noise schedule that preserves more information in high-content regions.
    """
    
    def __init__(self, sigma_min=1e-4, sigma_max=20.0, entropy_estimator=None):
        """
        Args:
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            entropy_estimator: Optional entropy estimator for adaptive scheduling
        """
        super().__init__()
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self.entropy_estimator = entropy_estimator
        
        # Learnable parameters for adaptive scheduling
        self.info_scale = nn.Parameter(torch.tensor(0.5))
        self.info_bias = nn.Parameter(torch.tensor(0.0))
        
    @property
    def sigma_min(self):
        return self._sigma_min
    
    @property 
    def sigma_max(self):
        return self._sigma_max
    
    def total_noise(self, t):
        """
        Total noise at time t (compatible with base Noise class).
        """
        # Ensure t is in [0, 1]
        t = torch.clamp(t, 0, 1)
        
        # Geometric interpolation in log space
        log_sigma = (1 - t) * np.log(self._sigma_min) + t * np.log(self._sigma_max)
        return torch.exp(log_sigma)
    
    def sigma(self, t):
        """Alias for total_noise for compatibility."""
        return self.total_noise(t)
    
    def adaptive_sigma(self, t, x_0):
        """
        Compute adaptive sigma based on information content of x_0.
        
        Args:
            t: Time step(s)
            x_0: Original data to estimate information content
            
        Returns:
            Modulated sigma values
        """
        # Get base sigma
        base_sigma = self.sigma(t)
        
        if self.entropy_estimator is None or x_0 is None:
            return base_sigma
            
        # Estimate information content
        with torch.no_grad():
            info_content = self.entropy_estimator.estimate_information_content(x_0)  # (B,)
            
        # Modulate sigma based on information content
        # High info content → preserve more → lower sigma
        # Low info content → can add more noise → higher sigma
        info_factor = 1.0 - torch.sigmoid(self.info_scale * info_content + self.info_bias)
        
        # Ensure info_factor has correct shape
        if info_factor.dim() == 1 and base_sigma.dim() == 0:
            info_factor = info_factor.mean()
        elif info_factor.dim() == 1 and base_sigma.dim() == 1:
            # Broadcast to match t's batch dimension
            if info_factor.shape[0] != base_sigma.shape[0]:
                info_factor = info_factor.mean()
                
        # Apply modulation
        adaptive_sigma = base_sigma * (0.5 + info_factor)
        
        # Ensure we stay within bounds
        adaptive_sigma = torch.clamp(adaptive_sigma, self._sigma_min, self._sigma_max)
        
        return adaptive_sigma
    
    def rate_noise(self, t):
        """Rate of change of noise (derivative)."""
        t = torch.clamp(t, 0, 1)
        
        # Derivative of geometric interpolation
        log_ratio = np.log(self._sigma_max / self._sigma_min)
        return self.total_noise(t) * log_ratio
    
    def dsigma_dt(self, t):
        """Alias for rate_noise for compatibility."""
        return self.rate_noise(t)


class ContentAwareNoise(InformationPreservingNoise):
    """
    Extended version that considers local content patterns.
    """
    
    def __init__(self, sigma_min=1e-4, sigma_max=20.0, entropy_estimator=None,
                 position_aware=True):
        super().__init__(sigma_min, sigma_max, entropy_estimator)
        self.position_aware = position_aware
        
        if position_aware:
            # Position-specific noise modulation
            self.position_scale = nn.Parameter(torch.ones(1024) * 0.1)
            
    def adaptive_sigma_per_position(self, t, x_0):
        """
        Compute position-specific adaptive sigma.
        
        Args:
            t: Time step
            x_0: Original tokens (batch_size, seq_len)
            
        Returns:
            sigma: (batch_size, seq_len) if position_aware else (batch_size,)
        """
        batch_size, seq_len = x_0.shape
        
        # Get base adaptive sigma
        base_sigma = self.adaptive_sigma(t, x_0)  # scalar or (B,)
        
        if not self.position_aware:
            return base_sigma
            
        # Get position-specific entropy
        if self.entropy_estimator is not None:
            with torch.no_grad():
                position_entropy = self.entropy_estimator(x_0, return_all=True)  # (B, L)
        else:
            # Fallback to uniform
            position_entropy = torch.ones(batch_size, seq_len, device=x_0.device)
            
        # Apply position-specific scaling
        position_factors = self.position_scale[:seq_len].unsqueeze(0)  # (1, L)
        position_modulation = 1.0 + position_factors * (position_entropy - 0.5)
        
        # Ensure base_sigma has correct shape for broadcasting
        if base_sigma.dim() == 0:
            base_sigma = base_sigma.unsqueeze(0).unsqueeze(0)
        elif base_sigma.dim() == 1:
            base_sigma = base_sigma.unsqueeze(1)
            
        # Apply position-specific modulation
        sigma_per_position = base_sigma * position_modulation
        
        return sigma_per_position


class LearnableNoise(Noise):
    """
    Fully learnable noise schedule that adapts during training.
    """
    
    def __init__(self, num_steps=1000, init_schedule='geometric'):
        super().__init__()
        self.num_steps = num_steps
        
        # Initialize with existing schedule
        if init_schedule == 'geometric':
            init_noise = GeometricNoise()
        else:
            init_noise = LogLinearNoise()
            
        # Create learnable parameters
        t_vals = torch.linspace(0, 1, num_steps)
        init_sigmas = torch.stack([init_noise.total_noise(t) for t in t_vals])
        
        # Log-space parameters for stability
        self.log_sigmas = nn.Parameter(torch.log(init_sigmas))
        
    @property
    def sigma_min(self):
        return torch.exp(self.log_sigmas.min())
    
    @property
    def sigma_max(self):
        return torch.exp(self.log_sigmas.max())
    
    def total_noise(self, t):
        """Interpolate learned sigma values."""
        # Ensure t is in [0, 1]
        t = torch.clamp(t, 0, 1)
        
        # Convert to indices
        indices = t * (self.num_steps - 1)
        
        # Linear interpolation
        idx_low = torch.floor(indices).long()
        idx_high = torch.ceil(indices).long()
        
        # Clamp indices
        idx_low = torch.clamp(idx_low, 0, self.num_steps - 1)
        idx_high = torch.clamp(idx_high, 0, self.num_steps - 1)
        
        # Get sigma values
        sigma_low = torch.exp(self.log_sigmas[idx_low])
        sigma_high = torch.exp(self.log_sigmas[idx_high])
        
        # Interpolation weights
        w = indices - idx_low.float()
        
        # Interpolate
        sigma = (1 - w) * sigma_low + w * sigma_high
        
        return sigma
    
    def sigma(self, t):
        """Alias for compatibility."""
        return self.total_noise(t)
    
    def rate_noise(self, t):
        """Approximate derivative using finite differences."""
        eps = 1e-4
        t_plus = torch.clamp(t + eps, 0, 1)
        t_minus = torch.clamp(t - eps, 0, 1)
        
        sigma_plus = self.total_noise(t_plus)
        sigma_minus = self.total_noise(t_minus)
        
        return (sigma_plus - sigma_minus) / (2 * eps)
    
    def dsigma_dt(self, t):
        """Alias for compatibility."""
        return self.rate_noise(t)
    
    def regularization_loss(self):
        """Regularization to ensure smooth noise schedule."""
        # Smoothness penalty
        sigmas = torch.exp(self.log_sigmas)
        smoothness = torch.mean((sigmas[1:] - sigmas[:-1]) ** 2)
        
        # Monotonicity penalty (should increase with t)
        monotone = torch.mean(torch.relu(sigmas[:-1] - sigmas[1:]))
        
        return smoothness + 10.0 * monotone