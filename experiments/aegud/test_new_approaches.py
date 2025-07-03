"""
Quick test script for new AEGUD approaches
"""

import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from noise_lib import GeometricNoise
from experiments.aegud.src.enhanced_adaptive_uniform_graph_v2 import EnhancedAdaptiveUniformV2
from experiments.aegud.src.hierarchical_diffusion import HierarchicalDiffusion
from experiments.aegud.src.enhanced_metrics import EnhancedMetrics

def test_enhanced_v2():
    """Test Enhanced AEGUD V2 variants."""
    device = 'cuda:0'
    vocab_size = 50
    batch_size = 8
    seq_len = 16
    
    print("Testing Enhanced AEGUD V2 Variants...")
    print("="*60)
    
    # Create noise
    noise = GeometricNoise(sigma_min=0.001, sigma_max=100.0)
    
    # Test configurations
    configs = {
        'Vocabulary-Aware': {
            'use_vocabulary_aware_decay': True,
            'use_learnable_schedule': False,
            'use_information_bottleneck': False
        },
        'Learnable Schedule': {
            'use_vocabulary_aware_decay': False,
            'use_learnable_schedule': True,
            'use_information_bottleneck': False
        },
        'Information Bottleneck': {
            'use_vocabulary_aware_decay': False,
            'use_learnable_schedule': False,
            'use_information_bottleneck': True
        },
        'Combined Features': {
            'use_vocabulary_aware_decay': True,
            'use_learnable_schedule': True,
            'use_information_bottleneck': True,
            'use_two_stage': True,
            'relaxed_convergence_epsilon': 0.1
        }
    }
    
    for name, config in configs.items():
        print(f"\nTesting {name}...")
        
        try:
            # Create graph
            graph = EnhancedAdaptiveUniformV2(
                dim=vocab_size,
                entropy_scale=1.0,
                **config
            ).to(device)
            
            # Test forward diffusion
            x_0 = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            
            for t_val in [0.1, 0.5, 0.9]:
                t = torch.tensor([t_val], device=device)
                sigma, _ = noise(t)
                
                # Test sample transition
                x_t = graph.sample_transition(x_0, sigma.expand(batch_size, 1))
                
                # Test adaptive weight
                adaptive_weight = graph.get_adaptive_weight(t_val, 1.0, x_t)
                
                # Test convergence metrics
                metrics = graph.get_convergence_metrics(x_t, t_val, 1.0)
                
                print(f"  t={t_val:.1f}: adaptive_weight={adaptive_weight:.3f}, "
                      f"entropy={metrics['entropy']:.3f}, kl={metrics['kl_from_uniform']:.3f}")
            
            # Test score entropy computation (simplified)
            from model.transformer import SEDD
            model_config = {
                'model': {
                    'hidden_size': 64,
                    'n_heads': 4,
                    'n_blocks': 2,
                    'cond_dim': 32,
                    'dropout': 0.1,
                    'scale_by_sigma': False
                },
                'tokens': vocab_size,
                'graph': {'type': 'uniform'}
            }
            model = SEDD(model_config).to(device)
            
            # Test training step
            t = torch.rand(batch_size, device=device)
            sigma, _ = noise(t)
            x_t = graph.sample_transition(x_0, sigma[:, None])
            
            # Model forward
            score = model(x_t, sigma)
            
            # Test loss computation
            if hasattr(graph, 'score_entropy_with_kl'):
                # Fix: ensure sigma has correct shape
                loss = graph.score_entropy_with_kl(
                    score, sigma[:, None], x_t, x_0, 
                    t=t.mean().item(), max_t=1.0
                )
                print(f"  Loss computation successful: {loss.mean().item():.4f}")
            
            print(f"  ✓ {name} test passed!")
            
        except Exception as e:
            print(f"  ✗ {name} test failed: {str(e)}")


def test_hierarchical_diffusion():
    """Test Hierarchical Diffusion."""
    device = 'cuda:0'
    vocab_size = 100
    
    print("\n\nTesting Hierarchical Diffusion...")
    print("="*60)
    
    try:
        # Create hierarchical diffusion
        hierarchical = HierarchicalDiffusion(
            vocab_size=vocab_size,
            num_levels=3,
            compression_ratios=[4, 2, 1]
        ).to(device)
        
        # Test encoding/decoding
        x_0 = torch.randint(0, vocab_size, (4, 32), device=device)
        
        # Test encoding to different levels
        for level in range(3):
            encoded = hierarchical.encode_to_level(x_0, level)
            decoded = hierarchical.decode_from_level(encoded, level)
            print(f"  Level {level}: encoded shape={encoded.shape}, "
                  f"vocab_size={hierarchical.level_vocab_sizes[level]}")
        
        # Test hierarchical diffusion step
        t = 0.5
        noise_scales = hierarchical.get_hierarchical_noise_schedule(t)
        x_t, level_outputs = hierarchical.hierarchical_diffusion_step(x_0, t, noise_scales)
        
        print(f"  Hierarchical diffusion step successful!")
        print(f"  Output shape: {x_t.shape}")
        
        # Validate convergence
        print("\n  Running convergence validation...")
        results = hierarchical.validate_hierarchical_convergence(
            num_steps=20, 
            device=device
        )
        
        print("  ✓ Hierarchical diffusion test passed!")
        
    except Exception as e:
        print(f"  ✗ Hierarchical diffusion test failed: {str(e)}")


def test_enhanced_metrics():
    """Test Enhanced Metrics."""
    device = 'cuda:0'
    vocab_size = 50
    
    print("\n\nTesting Enhanced Metrics...")
    print("="*60)
    
    try:
        metrics = EnhancedMetrics(vocab_size, device)
        
        # Test semantic preservation
        x_0 = torch.randint(0, vocab_size, (8, 32), device=device)
        x_t = x_0.clone()
        # Add some noise
        mask = torch.rand_like(x_t, dtype=torch.float) < 0.3
        x_t[mask] = torch.randint(0, vocab_size, mask.sum().item(), device=device)
        
        preservation = metrics.measure_semantic_preservation(x_0, x_t, t=0.5)
        print(f"  Semantic preservation metrics:")
        for k, v in preservation.items():
            print(f"    {k}: {v:.4f}")
        
        # Test diversity analysis
        samples = [torch.randint(0, vocab_size, (16, 32), device=device) for _ in range(5)]
        diversity = metrics.analyze_generation_diversity(samples)
        print(f"\n  Diversity metrics:")
        for k, v in diversity.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
        
        print("  ✓ Enhanced metrics test passed!")
        
    except Exception as e:
        print(f"  ✗ Enhanced metrics test failed: {str(e)}")


def test_relaxed_convergence():
    """Test relaxed convergence criteria."""
    device = 'cuda:0'
    vocab_size = 50
    
    print("\n\nTesting Relaxed Convergence...")
    print("="*60)
    
    try:
        # Create graph with relaxed convergence
        graph = EnhancedAdaptiveUniformV2(
            dim=vocab_size,
            relaxed_convergence_epsilon=0.15
        ).to(device)
        
        # Test convergence checking
        # Create nearly uniform distribution
        probs = torch.ones(vocab_size, device=device) / vocab_size
        probs[0] += 0.1 / vocab_size  # Small deviation
        probs = probs / probs.sum()  # Renormalize
        
        converged, max_dev = graph.check_relaxed_convergence(probs)
        print(f"  Nearly uniform: converged={converged}, max_deviation={max_dev:.6f}")
        
        # Create non-uniform distribution
        probs = torch.zeros(vocab_size, device=device)
        probs[:10] = 1.0 / 10  # Only first 10 tokens
        
        converged, max_dev = graph.check_relaxed_convergence(probs)
        print(f"  Non-uniform: converged={converged}, max_deviation={max_dev:.6f}")
        
        print("  ✓ Relaxed convergence test passed!")
        
    except Exception as e:
        print(f"  ✗ Relaxed convergence test failed: {str(e)}")


def main():
    print("Testing New AEGUD Research Directions")
    print("="*80)
    
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    test_enhanced_v2()
    test_hierarchical_diffusion()
    test_enhanced_metrics()
    test_relaxed_convergence()
    
    print("\n" + "="*80)
    print("All tests completed!")


if __name__ == "__main__":
    main()