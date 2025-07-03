#!/usr/bin/env python3
"""
Quick test to verify scaled experiment setup works correctly
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from run_final_scaled_experiments import AEGUDTrainer, get_experiment_config, set_seed

def test_scaled_setup():
    """Test the scaled experiment setup with minimal configuration."""
    print("Testing Enhanced AEGUD Scaled Experiment Setup")
    print("="*60)
    
    # Set seed
    set_seed(42)
    
    # Minimal test configuration
    base_config = {
        'vocab_size': 100,
        'seq_len': 32,
        'batch_size': 4,
        'use_wandb': False,
        'use_real_data': False,
        
        'model': {
            'hidden_size': 64,
            'n_heads': 4,
            'n_blocks': 2,
            'cond_dim': 32,
            'dropout': 0.1,
            'scale_by_sigma': False
        },
        
        'noise_type': 'geometric',
        'noise_params': {
            'sigma_min': 0.001,
            'sigma_max': 100.0
        },
        
        'optimizer': {
            'lr': 1e-3,
            'min_lr': 1e-6,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.01
        },
        
        'training': {
            'num_epochs': 1,
            'num_steps': 20,
            'batches_per_epoch': 10,
            'log_every': 5,
            'val_every': 10,
            'checkpoint_every': 15,
            'convergence_test_every': 15,
            'grad_clip': 1.0
        },
        
        'use_amp': False  # Disable for testing
    }
    
    # Test each experiment type
    experiments = [
        'baseline_uniform',
        'original_aegud', 
        'enhanced_v2_vocab_aware',
        'enhanced_v2_info_bottleneck',
        'enhanced_v2_full'
    ]
    
    for exp_name in experiments:
        print(f"\nTesting {exp_name}...")
        try:
            # Get config
            config = get_experiment_config(exp_name, base_config)
            
            # Create trainer
            trainer = AEGUDTrainer(config, device='cuda:0')
            
            # Test single training step
            batch = next(iter(trainer.train_loader))
            metrics = trainer.train_step(batch)
            
            print(f"  ✓ Training step successful, loss: {metrics['loss']:.4f}")
            
            # Test evaluation
            val_loss = trainer.evaluate(num_batches=2)
            print(f"  ✓ Evaluation successful, val_loss: {val_loss:.4f}")
            
            # Test convergence metrics (quick)
            convergence = trainer.metrics.measure_convergence_quality(
                trainer.graph, trainer.noise,
                num_steps=10,
                batch_size=4,
                seq_len=32
            )
            print(f"  ✓ Convergence test successful, final_kl: {convergence['kl_trajectory'][-1]:.4f}")
            
        except Exception as e:
            print(f"  ✗ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Setup test completed!")
    print("\nIf all tests passed, you can run the full experiments with:")
    print("  ./experiments/aegud/launch_scaled_experiments.sh sequential")
    print("or")
    print("  ./experiments/aegud/launch_scaled_experiments.sh parallel")


if __name__ == "__main__":
    test_scaled_setup()