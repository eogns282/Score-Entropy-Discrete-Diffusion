"""
Quick test of enhanced AEGUD to verify theoretical improvements.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from noise_lib import GeometricNoise
from experiments.aegud.src.adaptive_uniform_graph import AdaptiveUniform
from experiments.aegud.src.enhanced_adaptive_uniform_graph import EnhancedAdaptiveUniform, create_enhanced_aegud


def test_convergence_properties(device='cuda:0'):
    """Test convergence properties of different AEGUD configurations."""
    
    print("Testing convergence properties of Enhanced AEGUD")
    print("="*60)
    
    vocab_size = 50  # Smaller for faster testing
    seq_len = 32
    num_time_steps = 50  # More steps for better convergence
    
    # Create noise schedule with higher max sigma
    noise = GeometricNoise(sigma_min=0.001, sigma_max=50.0)
    
    # Test configurations
    configs = {
        'Original AEGUD': AdaptiveUniform(
            dim=vocab_size,
            entropy_scale=1.0,
            sparsity_k=25
        ).to(device),
        
        'AEGUD + Asymptotic': create_enhanced_aegud(
            vocab_size,
            entropy_scale=1.0,
            sparsity_k=25,
            use_asymptotic_guarantee=True,
            use_two_stage=False,
            use_controlled_decay=False,
            kl_regularization_weight=0.0
        ).to(device),
        
        'AEGUD + Two-Stage': create_enhanced_aegud(
            vocab_size,
            entropy_scale=1.0,
            sparsity_k=25,
            use_asymptotic_guarantee=False,
            use_two_stage=True,
            stage_transition_point=0.8,
            use_controlled_decay=False,
            kl_regularization_weight=0.0
        ).to(device),
        
        'AEGUD + All Features': create_enhanced_aegud(
            vocab_size,
            entropy_scale=1.0,
            sparsity_k=25,
            use_asymptotic_guarantee=True,
            use_two_stage=True,
            stage_transition_point=0.8,
            use_controlled_decay=True,
            decay_tau=0.1,
            kl_regularization_weight=0.05
        ).to(device)
    }
    
    results = {}
    
    for name, graph in configs.items():
        print(f"\nTesting {name}...")
        
        # Generate initial sequence
        x_0 = torch.randint(0, vocab_size, (1, seq_len), device=device)
        
        # Track entropy and KL divergence over time
        entropies = []
        kl_divs = []
        
        for step in range(num_time_steps + 1):
            t = step / num_time_steps
            sigma = noise.total_noise(torch.tensor(t, device=device))
            
            # Forward diffusion
            if step == 0:
                x_t = x_0.clone()
            else:
                x_t = graph.sample_transition(x_t, sigma.unsqueeze(0))
            
            # Compute entropy
            token_counts = torch.zeros(vocab_size, device=device)
            token_counts.scatter_add_(0, x_t.flatten(), torch.ones_like(x_t.flatten(), dtype=torch.float))
            token_probs = token_counts / seq_len
            entropy = -torch.sum(token_probs * torch.log(token_probs + 1e-10)) / np.log(vocab_size)
            entropies.append(entropy.item())
            
            # Compute KL from uniform
            uniform_probs = torch.ones_like(token_probs) / vocab_size
            kl = torch.sum(token_probs * torch.log((token_probs + 1e-10) / uniform_probs))
            kl_divs.append(kl.item())
            
            if step % 10 == 0:
                print(f"  t={t:.2f}: entropy={entropy:.4f}, KL={kl:.4f}")
        
        # Check if converged
        final_entropy = entropies[-1]
        final_kl = kl_divs[-1]
        converged = final_entropy > 0.95 and final_kl < 0.01
        
        results[name] = {
            'entropies': entropies,
            'kl_divs': kl_divs,
            'final_entropy': final_entropy,
            'final_kl': final_kl,
            'converged': converged
        }
        
        print(f"  Final: entropy={final_entropy:.4f}, KL={final_kl:.4f}")
        print(f"  Convergence: {'PASSED' if converged else 'FAILED'}")
    
    # Summary
    print("\n" + "="*60)
    print("CONVERGENCE SUMMARY")
    print("="*60)
    
    for name, result in results.items():
        status = "✓ CONVERGED" if result['converged'] else "✗ NOT CONVERGED"
        print(f"{name:30} {status}")
        print(f"  Final entropy: {result['final_entropy']:.4f}")
        print(f"  Final KL:      {result['final_kl']:.6f}")
    
    # Find best configuration
    converged_configs = [name for name, r in results.items() if r['converged']]
    if converged_configs:
        print(f"\nConverged configurations: {', '.join(converged_configs)}")
    else:
        print("\nWARNING: No configurations achieved proper convergence!")
        print("This might be due to:")
        print("  - Too few time steps (increase num_time_steps)")
        print("  - Insufficient noise scale (adjust sigma_max)")
        print("  - Small vocabulary size (increase vocab_size)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"experiments/aegud/results/quick_test_{timestamp}.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return results


def test_adaptive_weights():
    """Test the adaptive weight functions."""
    print("\nTesting adaptive weight decay functions")
    print("="*60)
    
    # Create enhanced graph
    graph = create_enhanced_aegud(
        vocab_size=100,
        use_asymptotic_guarantee=True,
        use_two_stage=True,
        stage_transition_point=0.8,
        use_controlled_decay=True,
        decay_tau=0.1
    )
    
    # Test weight at different time points
    time_points = [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    
    print("Time | Adaptive Weight")
    print("-" * 30)
    
    for t in time_points:
        weight = graph.get_adaptive_weight(t, max_t=1.0)
        if isinstance(weight, torch.Tensor):
            weight = weight.item()
        print(f"{t:4.2f} | {weight:15.6f}")
    
    print("\nObservations:")
    print("- Weight should decay from 1.0 to 0.0 as t approaches 1.0")
    print("- Two-stage transition at t=0.8 should show sharp drop")
    print("- Controlled decay provides smooth transition")


if __name__ == "__main__":
    # Run convergence test
    results = test_convergence_properties()
    
    # Test adaptive weights
    test_adaptive_weights()
    
    print("\nQuick test completed!")