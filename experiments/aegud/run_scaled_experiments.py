"""
Run large-scale experiments for Enhanced AEGUD with realistic settings.
This script runs comprehensive experiments to validate theoretical improvements at scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import wandb

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import SEDD components
from model.transformer import SEDD
from noise_lib import GeometricNoise, LogLinearNoise
from graph_lib import Uniform, Absorbing
# from data import get_dataloaders
# from utils import set_seed

def set_seed(seed):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Import AEGUD components
from experiments.aegud.src.adaptive_uniform_graph import AdaptiveUniform
from experiments.aegud.src.enhanced_adaptive_uniform_graph import EnhancedAdaptiveUniform, create_enhanced_aegud
from experiments.aegud.src.entropy_estimator import EntropyEstimator, AdaptiveTransitionMatrix
from experiments.aegud.src.simple_loss import SimpleLoss
from experiments.aegud.src.diffusion_validator import DiffusionValidator, run_comprehensive_validation


class ScaledTrainer:
    """Trainer for large-scale experiments with proper logging and checkpointing."""
    
    def __init__(self, model, graph, noise, loss_fn, optimizer_config, device='cuda'):
        self.model = model.to(device)
        self.graph = graph
        self.noise = noise
        self.loss_fn = loss_fn
        self.device = device
        
        # Get all trainable parameters
        params = list(self.model.parameters())
        if hasattr(self.graph, 'parameters'):
            params.extend(self.graph.parameters())
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            params,
            lr=optimizer_config['lr'],
            betas=optimizer_config['betas'],
            eps=optimizer_config['eps'],
            weight_decay=optimizer_config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=optimizer_config['max_steps'],
            eta_min=optimizer_config['min_lr']
        )
        
        # Training state
        self.step = 0
        self.best_loss = float('inf')
        
    def train_step(self, batch):
        """Single training step with gradient accumulation support."""
        self.model.train()
        
        # Sample time uniformly
        batch_size = batch.shape[0]
        t = torch.rand(batch_size, device=self.device)
        
        # Get noise values
        sigma, dsigma = self.noise(t)
        
        # Forward diffusion
        x_t = self.graph.sample_transition(batch, sigma[:, None])
        
        # Model forward pass
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            score = self.model(x_t, sigma)
        
        # Compute loss
        if hasattr(self.graph, 'score_entropy_with_kl'):
            # Enhanced graph with KL regularization
            loss = self.loss_fn(score, x_t, batch, sigma, self.graph, t=t.mean())
        else:
            loss = self.loss_fn(score, x_t, batch, sigma, self.graph)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        self.step += 1
        
        return {
            'loss': loss.item(),
            'lr': self.scheduler.get_last_lr()[0],
            'sigma_mean': sigma.mean().item()
        }
    
    def evaluate(self, val_loader, num_batches=10):
        """Evaluate model on validation set."""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= num_batches:
                    break
                
                batch = batch.to(self.device)
                t = torch.rand(batch.shape[0], device=self.device)
                sigma, _ = self.noise(t)
                
                x_t = self.graph.sample_transition(batch, sigma[:, None])
                score = self.model(x_t, sigma)
                
                if hasattr(self.graph, 'score_entropy_with_kl'):
                    loss = self.loss_fn(score, x_t, batch, sigma, self.graph, t=t.mean())
                else:
                    loss = self.loss_fn(score, x_t, batch, sigma, self.graph)
                
                val_losses.append(loss.item())
        
        return np.mean(val_losses)
    
    def save_checkpoint(self, path, metrics=None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'best_loss': self.best_loss,
            'metrics': metrics
        }
        
        if hasattr(self.graph, 'state_dict'):
            checkpoint['graph_state_dict'] = self.graph.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.best_loss = checkpoint['best_loss']
        
        if hasattr(self.graph, 'load_state_dict') and 'graph_state_dict' in checkpoint:
            self.graph.load_state_dict(checkpoint['graph_state_dict'])
        
        return checkpoint.get('metrics', {})


def create_scaled_model(vocab_size, config):
    """Create a properly sized model for scaled experiments."""
    model_config = {
        'model': {
            'hidden_size': config['hidden_size'],
            'n_heads': config['n_heads'],
            'n_blocks': config['n_blocks'],
            'cond_dim': config['cond_dim'],
            'dropout': config['dropout'],
            'scale_by_sigma': config.get('scale_by_sigma', False)
        },
        'tokens': vocab_size,
        'graph': {
            'type': 'uniform'  # Will be overridden
        }
    }
    
    return SEDD(model_config)


def run_scaled_experiment(experiment_name, config, device='cuda'):
    """Run a single scaled experiment configuration."""
    print(f"\n{'='*80}")
    print(f"Running Scaled Experiment: {experiment_name}")
    print(f"{'='*80}")
    
    # Set random seed
    set_seed(config['seed'])
    
    # Create directories
    exp_dir = Path(f"experiments/aegud/scaled_results/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Initialize wandb if requested
    if config.get('use_wandb', False):
        wandb.init(
            project="enhanced-aegud",
            name=experiment_name,
            config=config
        )
    
    # Create model
    model = create_scaled_model(config['vocab_size'], config['model'])
    
    # Create graph based on experiment type
    if config['graph_type'] == 'original_aegud':
        graph = AdaptiveUniform(
            dim=config['vocab_size'],
            entropy_scale=config['entropy_scale'],
            sparsity_k=config.get('sparsity_k', None)
        ).to(device)
    elif config['graph_type'] == 'enhanced_aegud':
        graph = create_enhanced_aegud(
            config['vocab_size'],
            entropy_scale=config['entropy_scale'],
            sparsity_k=config.get('sparsity_k', None),
            **config['enhancement_params']
        ).to(device)
    elif config['graph_type'] == 'uniform':
        graph = Uniform(config['vocab_size'])
    elif config['graph_type'] == 'absorbing':
        graph = Absorbing(config['vocab_size'])
    else:
        raise ValueError(f"Unknown graph type: {config['graph_type']}")
    
    # Create noise schedule
    if config['noise_type'] == 'geometric':
        noise = GeometricNoise(
            sigma_min=config['noise_params']['sigma_min'],
            sigma_max=config['noise_params']['sigma_max']
        )
    else:
        noise = LogLinearNoise(eps=config['noise_params'].get('eps', 1e-3))
    
    # Create loss function
    if hasattr(graph, 'score_entropy_with_kl'):
        def loss_fn(score, x_t, x_0, sigma, graph, t=None):
            return graph.score_entropy_with_kl(score, sigma, x_t, x_0, t=t, max_t=1.0).mean()
    else:
        loss_fn = SimpleLoss()
    
    # Create data loaders (using synthetic data for now)
    train_loader = create_synthetic_dataloader(
        vocab_size=config['vocab_size'],
        seq_len=config['seq_len'],
        batch_size=config['batch_size'],
        num_batches=config['train_batches_per_epoch']
    )
    
    val_loader = create_synthetic_dataloader(
        vocab_size=config['vocab_size'],
        seq_len=config['seq_len'],
        batch_size=config['batch_size'],
        num_batches=50
    )
    
    # Create trainer
    trainer = ScaledTrainer(
        model, graph, noise, loss_fn,
        optimizer_config=config['optimizer'],
        device=device
    )
    
    # Training metrics
    metrics = defaultdict(list)
    best_val_loss = float('inf')
    
    # Training loop
    print(f"\nStarting training for {config['num_steps']} steps...")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            
            # Train step
            step_metrics = trainer.train_step(batch)
            epoch_losses.append(step_metrics['loss'])
            
            # Log metrics
            if trainer.step % config['log_every'] == 0:
                avg_loss = np.mean(epoch_losses[-100:])
                elapsed = time.time() - start_time
                steps_per_sec = trainer.step / elapsed
                
                print(f"Step {trainer.step}: loss={avg_loss:.4f}, "
                      f"lr={step_metrics['lr']:.6f}, "
                      f"speed={steps_per_sec:.2f} steps/s")
                
                metrics['train_loss'].append(avg_loss)
                metrics['learning_rate'].append(step_metrics['lr'])
                metrics['steps'].append(trainer.step)
                
                if config.get('use_wandb', False):
                    wandb.log({
                        'train_loss': avg_loss,
                        'learning_rate': step_metrics['lr'],
                        'steps_per_second': steps_per_sec,
                        'step': trainer.step
                    })
            
            # Validation
            if trainer.step % config['val_every'] == 0:
                val_loss = trainer.evaluate(val_loader)
                metrics['val_loss'].append(val_loss)
                metrics['val_steps'].append(trainer.step)
                
                print(f"Validation at step {trainer.step}: loss={val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    trainer.save_checkpoint(
                        checkpoint_dir / 'best_model.pt',
                        metrics={'val_loss': val_loss, 'step': trainer.step}
                    )
                
                if config.get('use_wandb', False):
                    wandb.log({'val_loss': val_loss, 'step': trainer.step})
            
            # Convergence validation
            if trainer.step % config['convergence_test_every'] == 0:
                print(f"\nRunning convergence validation at step {trainer.step}...")
                validator = DiffusionValidator(config['vocab_size'], device)
                
                convergence_results = validator.test_forward_diffusion_convergence(
                    graph, noise,
                    num_steps=50,
                    batch_size=16,
                    seq_len=config['seq_len']
                )
                
                conv_metrics = convergence_results['convergence']
                metrics['convergence_tests'].append({
                    'step': trainer.step,
                    'final_entropy': convergence_results['metrics']['entropy'][-1],
                    'final_kl': convergence_results['metrics']['kl_from_uniform'][-1],
                    'converged': conv_metrics['converged']
                })
                
                print(f"Convergence: {'PASSED' if conv_metrics['converged'] else 'FAILED'}")
                print(f"  Final entropy: {conv_metrics['entropy_ratio']:.4f}")
                print(f"  Final KL: {convergence_results['metrics']['kl_from_uniform'][-1]:.6f}")
                
                if config.get('use_wandb', False):
                    wandb.log({
                        'convergence_entropy': conv_metrics['entropy_ratio'],
                        'convergence_kl': convergence_results['metrics']['kl_from_uniform'][-1],
                        'convergence_passed': int(conv_metrics['converged']),
                        'step': trainer.step
                    })
            
            # Save periodic checkpoint
            if trainer.step % config['checkpoint_every'] == 0:
                trainer.save_checkpoint(
                    checkpoint_dir / f'checkpoint_step_{trainer.step}.pt',
                    metrics={'train_loss': avg_loss, 'step': trainer.step}
                )
            
            # Check if we've reached max steps
            if trainer.step >= config['num_steps']:
                break
        
        if trainer.step >= config['num_steps']:
            break
    
    # Final evaluation
    print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
    print("Running final comprehensive validation...")
    
    # Load best model
    trainer.load_checkpoint(checkpoint_dir / 'best_model.pt')
    
    # Run comprehensive validation
    final_validation = run_comprehensive_validation(
        graph, noise, model,
        vocab_size=config['vocab_size'],
        device=device
    )
    
    # Save all results
    results = {
        'experiment_name': experiment_name,
        'config': config,
        'metrics': dict(metrics),
        'final_validation': final_validation,
        'best_val_loss': best_val_loss,
        'training_time': time.time() - start_time
    }
    
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {exp_dir}")
    
    if config.get('use_wandb', False):
        wandb.finish()
    
    return results


def create_synthetic_dataloader(vocab_size, seq_len, batch_size, num_batches):
    """Create synthetic data loader for experiments."""
    class SyntheticDataset(torch.utils.data.Dataset):
        def __init__(self, vocab_size, seq_len, num_samples):
            self.vocab_size = vocab_size
            self.seq_len = seq_len
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Generate diverse synthetic sequences
            if idx % 4 == 0:
                # Repeated pattern
                pattern_len = max(1, self.seq_len // 4)
                pattern = torch.randint(0, self.vocab_size, (pattern_len,))
                repeats = (self.seq_len + pattern_len - 1) // pattern_len
                seq = pattern.repeat(repeats)[:self.seq_len]
            elif idx % 4 == 1:
                # Gradual transition
                seq = torch.linspace(0, self.vocab_size-1, self.seq_len).long()
            elif idx % 4 == 2:
                # Clustered tokens
                num_clusters = 5
                cluster_size = self.seq_len // num_clusters
                remainder = self.seq_len % num_clusters
                
                clusters = []
                for i in range(num_clusters):
                    size = cluster_size + (1 if i < remainder else 0)
                    clusters.append(torch.full((size,), i % self.vocab_size))
                seq = torch.cat(clusters)
            else:
                # Random
                seq = torch.randint(0, self.vocab_size, (self.seq_len,))
            
            # Ensure correct length
            assert seq.shape[0] == self.seq_len, f"Sequence length mismatch: {seq.shape[0]} != {self.seq_len}"
            
            return seq
    
    dataset = SyntheticDataset(vocab_size, seq_len, batch_size * num_batches)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )


def run_all_scaled_experiments(base_config, device='cuda'):
    """Run all experiment configurations at scale."""
    
    # Define experiment configurations
    experiments = {
        'Original_AEGUD_Scaled': {
            **base_config,
            'graph_type': 'original_aegud',
            'entropy_scale': 1.0,
            'enhancement_params': {}
        },
        
        'Enhanced_AEGUD_Asymptotic': {
            **base_config,
            'graph_type': 'enhanced_aegud',
            'entropy_scale': 1.0,
            'enhancement_params': {
                'use_asymptotic_guarantee': True,
                'use_two_stage': False,
                'use_controlled_decay': False,
                'kl_regularization_weight': 0.0
            }
        },
        
        'Enhanced_AEGUD_TwoStage': {
            **base_config,
            'graph_type': 'enhanced_aegud',
            'entropy_scale': 1.0,
            'enhancement_params': {
                'use_asymptotic_guarantee': False,
                'use_two_stage': True,
                'stage_transition_point': 0.8,
                'use_controlled_decay': False,
                'kl_regularization_weight': 0.0
            }
        },
        
        'Enhanced_AEGUD_ControlledDecay': {
            **base_config,
            'graph_type': 'enhanced_aegud',
            'entropy_scale': 1.0,
            'enhancement_params': {
                'use_asymptotic_guarantee': False,
                'use_two_stage': False,
                'use_controlled_decay': True,
                'decay_tau': 0.1,
                'kl_regularization_weight': 0.0
            }
        },
        
        'Enhanced_AEGUD_KLReg': {
            **base_config,
            'graph_type': 'enhanced_aegud',
            'entropy_scale': 1.0,
            'enhancement_params': {
                'use_asymptotic_guarantee': False,
                'use_two_stage': False,
                'use_controlled_decay': False,
                'kl_regularization_weight': 0.1
            }
        },
        
        'Enhanced_AEGUD_Full': {
            **base_config,
            'graph_type': 'enhanced_aegud',
            'entropy_scale': 1.0,
            'enhancement_params': {
                'use_asymptotic_guarantee': True,
                'use_two_stage': True,
                'stage_transition_point': 0.8,
                'use_controlled_decay': True,
                'decay_tau': 0.1,
                'kl_regularization_weight': 0.05
            }
        },
        
        'Baseline_Uniform': {
            **base_config,
            'graph_type': 'uniform',
            'enhancement_params': {}
        },
        
        'Baseline_Absorbing': {
            **base_config,
            'graph_type': 'absorbing',
            'noise_type': 'loglinear',
            'model': {
                **base_config['model'],
                'scale_by_sigma': True
            },
            'enhancement_params': {}
        }
    }
    
    all_results = {}
    
    for exp_name, exp_config in experiments.items():
        print(f"\n{'#'*80}")
        print(f"# Experiment: {exp_name}")
        print(f"{'#'*80}")
        
        try:
            results = run_scaled_experiment(exp_name, exp_config, device)
            all_results[exp_name] = results
            
            # Save intermediate results
            with open(f"experiments/aegud/scaled_results/all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            
        except Exception as e:
            print(f"Error in experiment {exp_name}: {str(e)}")
            all_results[exp_name] = {'error': str(e)}
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run scaled Enhanced AEGUD experiments')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for training')
    parser.add_argument('--vocab_size', type=int, default=1000,
                       help='Vocabulary size')
    parser.add_argument('--num_steps', type=int, default=10000,
                       help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--experiment', type=str, default='all',
                       help='Which experiment to run (all or specific name)')
    
    args = parser.parse_args()
    
    # Base configuration for all experiments
    base_config = {
        'seed': 42,
        'vocab_size': args.vocab_size,
        'seq_len': 128,
        'batch_size': args.batch_size,
        'num_steps': args.num_steps,
        'num_epochs': 100,  # Max epochs (will stop at num_steps)
        'train_batches_per_epoch': 1000,
        
        'model': {
            'hidden_size': 512,
            'n_heads': 8,
            'n_blocks': 12,
            'cond_dim': 256,
            'dropout': 0.1,
            'scale_by_sigma': False
        },
        
        'noise_type': 'geometric',
        'noise_params': {
            'sigma_min': 0.001,
            'sigma_max': 100.0
        },
        
        'optimizer': {
            'lr': 3e-4,
            'min_lr': 1e-6,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.01,
            'max_steps': args.num_steps
        },
        
        'log_every': 50,
        'val_every': 500,
        'checkpoint_every': 2000,
        'convergence_test_every': 1000,
        
        'use_wandb': args.use_wandb
    }
    
    print(f"Running scaled experiments on {args.device}")
    print(f"Configuration:")
    print(f"  - Vocabulary size: {args.vocab_size}")
    print(f"  - Training steps: {args.num_steps}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Model size: ~65M parameters")
    
    # Create results directory
    os.makedirs("experiments/aegud/scaled_results", exist_ok=True)
    
    if args.experiment == 'all':
        # Run all experiments
        results = run_all_scaled_experiments(base_config, device=args.device)
    else:
        # Run specific experiment
        exp_config = {
            **base_config,
            'graph_type': 'enhanced_aegud' if 'enhanced' in args.experiment.lower() else 'original_aegud',
            'entropy_scale': 1.0,
            'enhancement_params': {
                'use_asymptotic_guarantee': True,
                'use_two_stage': True,
                'stage_transition_point': 0.8,
                'use_controlled_decay': True,
                'decay_tau': 0.1,
                'kl_regularization_weight': 0.05
            }
        }
        results = {args.experiment: run_scaled_experiment(args.experiment, exp_config, device=args.device)}
    
    # Create final summary
    print("\n" + "="*80)
    print("SCALED EXPERIMENT SUMMARY")
    print("="*80)
    
    for exp_name, result in results.items():
        if 'error' in result:
            print(f"\n{exp_name}: ERROR - {result['error']}")
        else:
            val = result.get('final_validation', {})
            conv = val.get('convergence', {}).get('convergence', {})
            
            print(f"\n{exp_name}:")
            print(f"  - Best Val Loss: {result.get('best_val_loss', 'N/A'):.4f}")
            print(f"  - Final Convergence: {'PASSED' if conv.get('converged', False) else 'FAILED'}")
            print(f"  - Final Entropy: {conv.get('entropy_ratio', 'N/A'):.4f}")
            print(f"  - Final KL: {val.get('convergence', {}).get('final_kl', 'N/A'):.6f}")
            print(f"  - Training Time: {result.get('training_time', 0)/3600:.2f} hours")
    
    # Save final summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f"experiments/aegud/scaled_results/final_summary_{timestamp}.json"
    
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nFinal summary saved to {summary_path}")


if __name__ == "__main__":
    main()