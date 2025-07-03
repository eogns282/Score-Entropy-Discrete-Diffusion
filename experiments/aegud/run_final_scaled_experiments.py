#!/usr/bin/env python3
"""
Final Scaled Experiments for Enhanced AEGUD V2
Ready for production runs with all fixes and optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import wandb

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import SEDD components
from model.transformer import SEDD
from noise_lib import GeometricNoise, LogLinearNoise
from graph_lib import Uniform
from data import get_dataloaders
# Define set_seed since it's not in utils
def set_seed(seed):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Import AEGUD components
from experiments.aegud.src.adaptive_uniform_graph import AdaptiveUniform
from experiments.aegud.src.enhanced_adaptive_uniform_graph import EnhancedAdaptiveUniform
from experiments.aegud.src.enhanced_adaptive_uniform_graph_v2 import EnhancedAdaptiveUniformV2
from experiments.aegud.src.entropy_estimator import EntropyEstimator, AdaptiveTransitionMatrix
from experiments.aegud.src.simple_loss import SimpleLoss
from experiments.aegud.src.enhanced_metrics import EnhancedMetrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'aegud_experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class AEGUDTrainer:
    """Production-ready trainer for AEGUD experiments."""
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.setup_experiment()
        
    def setup_experiment(self):
        """Setup model, graph, optimizer, and data."""
        logger.info(f"Setting up experiment: {self.config['experiment_name']}")
        
        # Create model
        self.model = self.create_model()
        
        # Create graph
        self.graph = self.create_graph()
        
        # Create noise schedule
        self.noise = self.create_noise()
        
        # Create optimizer
        self.setup_optimizer()
        
        # Create data loaders
        self.setup_data()
        
        # Create metrics tracker
        self.metrics = EnhancedMetrics(self.config['vocab_size'], self.device)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
    def create_model(self):
        """Create SEDD model with appropriate configuration."""
        model_config = {
            'model': self.config['model'],
            'tokens': self.config['vocab_size'],
            'graph': {'type': 'uniform'}  # Will use custom graph
        }
        
        model = SEDD(model_config).to(self.device)
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
        
        return model
    
    def create_graph(self):
        """Create appropriate graph based on configuration."""
        graph_type = self.config['graph_type']
        vocab_size = self.config['vocab_size']
        
        logger.info(f"Creating graph: {graph_type}")
        
        if graph_type == 'baseline_uniform':
            return Uniform(vocab_size)
            
        elif graph_type == 'original_aegud':
            return AdaptiveUniform(
                dim=vocab_size,
                entropy_scale=self.config.get('entropy_scale', 1.0)
            ).to(self.device)
            
        elif graph_type == 'enhanced_v2_vocab_aware':
            return EnhancedAdaptiveUniformV2(
                dim=vocab_size,
                entropy_scale=self.config.get('entropy_scale', 1.0),
                use_vocabulary_aware_decay=True,
                use_two_stage=True,
                stage_transition_point=0.8,
                relaxed_convergence_epsilon=0.1
            ).to(self.device)
            
        elif graph_type == 'enhanced_v2_info_bottleneck':
            return EnhancedAdaptiveUniformV2(
                dim=vocab_size,
                entropy_scale=self.config.get('entropy_scale', 1.0),
                use_information_bottleneck=True,
                info_bottleneck_beta=0.1,
                relaxed_convergence_epsilon=0.1
            ).to(self.device)
            
        elif graph_type == 'enhanced_v2_full':
            graph = EnhancedAdaptiveUniformV2(
                dim=vocab_size,
                entropy_scale=self.config.get('entropy_scale', 1.0),
                use_vocabulary_aware_decay=True,
                use_learnable_schedule=True,
                use_information_bottleneck=True,
                use_two_stage=True,
                stage_transition_point=0.8,
                use_controlled_decay=True,
                decay_tau=0.1,
                kl_regularization_weight=0.05,
                relaxed_convergence_epsilon=0.1
            ).to(self.device)
            
            # Fix device placement for learnable schedule
            if hasattr(graph, 'learnable_schedule'):
                graph.learnable_schedule = graph.learnable_schedule.to(self.device)
            
            return graph
            
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
    
    def create_noise(self):
        """Create noise schedule."""
        if self.config['noise_type'] == 'geometric':
            return GeometricNoise(
                sigma_min=self.config['noise_params']['sigma_min'],
                sigma_max=self.config['noise_params']['sigma_max']
            )
        else:
            return LogLinearNoise(eps=self.config['noise_params'].get('eps', 1e-3))
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Collect all parameters
        params = list(self.model.parameters())
        if hasattr(self.graph, 'parameters'):
            params.extend(self.graph.parameters())
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config['optimizer']['lr'],
            betas=self.config['optimizer']['betas'],
            eps=self.config['optimizer']['eps'],
            weight_decay=self.config['optimizer']['weight_decay']
        )
        
        # Create scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['num_steps'],
            eta_min=self.config['optimizer']['min_lr']
        )
        
        logger.info("Optimizer and scheduler created")
    
    def setup_data(self):
        """Setup data loaders."""
        if self.config.get('use_real_data', False):
            # Use real WikiText data
            train_loader, val_loader = get_dataloaders(
                batch_size=self.config['batch_size'],
                dataset_name='wikitext103',
                tokenizer_name='gpt2'
            )
            self.train_loader = train_loader
            self.val_loader = val_loader
        else:
            # Use synthetic data for testing
            self.train_loader = self.create_synthetic_loader(
                num_batches=self.config['training']['batches_per_epoch']
            )
            self.val_loader = self.create_synthetic_loader(num_batches=50)
        
        logger.info(f"Data loaders created: {len(self.train_loader)} train batches")
    
    def create_synthetic_loader(self, num_batches):
        """Create synthetic data loader."""
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, vocab_size, seq_len, num_samples):
                self.vocab_size = vocab_size
                self.seq_len = seq_len
                self.num_samples = num_samples
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                # Create diverse synthetic patterns
                pattern_type = idx % 5
                if pattern_type == 0:
                    # Random sequence
                    return torch.randint(0, self.vocab_size, (self.seq_len,))
                elif pattern_type == 1:
                    # Repeated pattern
                    pattern = torch.randint(0, self.vocab_size, (self.seq_len // 4,))
                    return pattern.repeat(4)[:self.seq_len]
                elif pattern_type == 2:
                    # Gradual transition
                    return torch.linspace(0, self.vocab_size-1, self.seq_len).long()
                elif pattern_type == 3:
                    # Clustered tokens
                    clusters = []
                    for i in range(4):
                        cluster_val = (i * self.vocab_size) // 4
                        clusters.append(torch.full((self.seq_len // 4,), cluster_val))
                    return torch.cat(clusters)[:self.seq_len]
                else:
                    # Sparse sequence
                    seq = torch.zeros(self.seq_len, dtype=torch.long)
                    indices = torch.randperm(self.seq_len)[:self.seq_len // 3]
                    seq[indices] = torch.randint(1, self.vocab_size, (len(indices),))
                    return seq
        
        dataset = SyntheticDataset(
            self.config['vocab_size'],
            self.config['seq_len'],
            self.config['batch_size'] * num_batches
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
    
    def train_step(self, batch):
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        batch = batch.to(self.device)
        batch_size = batch.shape[0]
        
        # Sample time uniformly
        t = torch.rand(batch_size, device=self.device)
        
        # Get noise
        sigma, dsigma = self.noise(t)
        
        # Forward diffusion
        x_t = self.graph.sample_transition(batch, sigma[:, None])
        
        # Ensure x_t is within vocabulary bounds (fix for index error)
        x_t = torch.clamp(x_t, 0, self.config['vocab_size'] - 1)
        
        # Model forward pass
        with torch.cuda.amp.autocast(enabled=self.config.get('use_amp', True)):
            score = self.model(x_t, sigma)
        
        # Compute loss
        if hasattr(self.graph, 'score_entropy_with_kl'):
            # Ensure sigma has correct shape
            loss = self.graph.score_entropy_with_kl(
                score, sigma[:, None], x_t, batch,
                t=t.mean().item(), max_t=1.0
            )
            loss = loss.mean()
        else:
            loss = self.graph.score_entropy(score, sigma[:, None], x_t, batch).mean()
            loss = (dsigma[:, None] * loss).sum(dim=-1).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config['training'].get('grad_clip', 1.0)
        )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        # Update global step
        self.global_step += 1
        
        # Collect metrics
        metrics = {
            'loss': loss.item(),
            'lr': self.scheduler.get_last_lr()[0],
            'grad_norm': grad_norm.item(),
            'sigma_mean': sigma.mean().item()
        }
        
        # Add graph-specific metrics
        if hasattr(self.graph, 'last_kl_loss'):
            metrics['kl_loss'] = self.graph.last_kl_loss
        
        return metrics
    
    def evaluate(self, num_batches=None):
        """Evaluate on validation set."""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if num_batches and i >= num_batches:
                    break
                
                batch = batch.to(self.device)
                batch_size = batch.shape[0]
                
                # Sample time
                t = torch.rand(batch_size, device=self.device)
                sigma, _ = self.noise(t)
                
                # Forward diffusion
                x_t = self.graph.sample_transition(batch, sigma[:, None])
                x_t = torch.clamp(x_t, 0, self.config['vocab_size'] - 1)
                
                # Model forward
                score = self.model(x_t, sigma)
                
                # Compute loss
                if hasattr(self.graph, 'score_entropy_with_kl'):
                    loss = self.graph.score_entropy_with_kl(
                        score, sigma[:, None], x_t, batch,
                        t=t.mean().item(), max_t=1.0
                    ).mean()
                else:
                    loss = self.graph.score_entropy(
                        score, sigma[:, None], x_t, batch
                    ).mean()
                
                val_losses.append(loss.item())
        
        return np.mean(val_losses)
    
    def save_checkpoint(self, path, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if hasattr(self.graph, 'state_dict'):
            checkpoint['graph_state_dict'] = self.graph.state_dict()
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.parent / 'best_model.pt'
            torch.save(checkpoint, best_path)
        
        logger.info(f"Checkpoint saved: {path}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Setup directories
        exp_dir = Path(f"experiments/aegud/results/{self.config['experiment_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        exp_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = exp_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize wandb if requested
        if self.config.get('use_wandb', False):
            wandb.init(
                project='enhanced-aegud',
                name=self.config['experiment_name'],
                config=self.config
            )
        
        # Training loop
        train_losses = []
        start_time = time.time()
        
        for epoch in range(self.config['training']['num_epochs']):
            self.epoch = epoch
            epoch_losses = []
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Training step
                metrics = self.train_step(batch)
                epoch_losses.append(metrics['loss'])
                train_losses.append(metrics['loss'])
                
                # Logging
                if self.global_step % self.config['training']['log_every'] == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = self.global_step / elapsed
                    avg_loss = np.mean(train_losses[-100:])
                    
                    logger.info(
                        f"Step {self.global_step} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {metrics['lr']:.6f} | "
                        f"Speed: {steps_per_sec:.2f} steps/s"
                    )
                    
                    if self.config.get('use_wandb', False):
                        wandb.log({
                            'train_loss': avg_loss,
                            'learning_rate': metrics['lr'],
                            'grad_norm': metrics['grad_norm'],
                            'steps_per_second': steps_per_sec,
                            'step': self.global_step
                        })
                
                # Validation
                if self.global_step % self.config['training']['val_every'] == 0:
                    val_loss = self.evaluate(num_batches=50)
                    
                    logger.info(f"Validation loss: {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(
                            checkpoint_dir / f'checkpoint_step_{self.global_step}.pt',
                            is_best=True
                        )
                    
                    if self.config.get('use_wandb', False):
                        wandb.log({
                            'val_loss': val_loss,
                            'step': self.global_step
                        })
                
                # Convergence test
                if self.global_step % self.config['training']['convergence_test_every'] == 0:
                    convergence_metrics = self.metrics.measure_convergence_quality(
                        self.graph, self.noise,
                        num_steps=50,
                        batch_size=16,
                        seq_len=self.config['seq_len']
                    )
                    
                    logger.info(
                        f"Convergence test: "
                        f"Entropy={convergence_metrics['entropy_trajectory'][-1]:.3f}, "
                        f"KL={convergence_metrics['kl_trajectory'][-1]:.4f}, "
                        f"Converged={convergence_metrics['final_status']['overall_converged']}"
                    )
                    
                    # Save convergence plot
                    self.metrics.create_convergence_visualization(
                        convergence_metrics,
                        exp_dir / f'convergence_step_{self.global_step}.png'
                    )
                
                # Regular checkpoint
                if self.global_step % self.config['training']['checkpoint_every'] == 0:
                    self.save_checkpoint(
                        checkpoint_dir / f'checkpoint_step_{self.global_step}.pt'
                    )
                
                # Check if done
                if self.global_step >= self.config['training']['num_steps']:
                    logger.info("Reached maximum steps, stopping training")
                    break
            
            # End of epoch logging
            epoch_avg_loss = np.mean(epoch_losses)
            logger.info(f"Epoch {epoch} completed | Avg loss: {epoch_avg_loss:.4f}")
            
            if self.global_step >= self.config['training']['num_steps']:
                break
        
        # Training completed
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/3600:.2f} hours")
        
        # Save final results
        results = {
            'experiment_name': self.config['experiment_name'],
            'config': self.config,
            'final_train_loss': np.mean(train_losses[-100:]),
            'best_val_loss': self.best_val_loss,
            'total_steps': self.global_step,
            'training_time': total_time
        }
        
        with open(exp_dir / 'final_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        if self.config.get('use_wandb', False):
            wandb.finish()
        
        return results


def get_experiment_config(experiment_name, base_config):
    """Get specific experiment configuration."""
    configs = {
        'enhanced_v2_vocab_aware': {
            'experiment_name': 'Enhanced_V2_VocabAware',
            'graph_type': 'enhanced_v2_vocab_aware',
            'entropy_scale': 1.0
        },
        'enhanced_v2_info_bottleneck': {
            'experiment_name': 'Enhanced_V2_InfoBottleneck',
            'graph_type': 'enhanced_v2_info_bottleneck',
            'entropy_scale': 1.0
        },
        'enhanced_v2_full': {
            'experiment_name': 'Enhanced_V2_Full',
            'graph_type': 'enhanced_v2_full',
            'entropy_scale': 1.0
        },
        'baseline_uniform': {
            'experiment_name': 'Baseline_Uniform',
            'graph_type': 'baseline_uniform'
        },
        'original_aegud': {
            'experiment_name': 'Original_AEGUD',
            'graph_type': 'original_aegud',
            'entropy_scale': 1.0
        }
    }
    
    if experiment_name not in configs:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    # Merge with base config
    config = base_config.copy()
    config.update(configs[experiment_name])
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Run scaled Enhanced AEGUD experiments')
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['enhanced_v2_vocab_aware', 'enhanced_v2_info_bottleneck', 
                               'enhanced_v2_full', 'baseline_uniform', 'original_aegud'],
                       help='Which experiment to run')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--use_real_data', action='store_true',
                       help='Use real WikiText data instead of synthetic')
    
    # Override default config values
    parser.add_argument('--vocab_size', type=int, default=5000,
                       help='Vocabulary size')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--num_steps', type=int, default=100000,
                       help='Number of training steps')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Base configuration
    base_config = {
        'vocab_size': args.vocab_size,
        'seq_len': 128,
        'batch_size': args.batch_size,
        'use_wandb': args.use_wandb,
        'use_real_data': args.use_real_data,
        
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
            'weight_decay': 0.01
        },
        
        'training': {
            'num_epochs': 100,
            'num_steps': args.num_steps,
            'batches_per_epoch': 1000,
            'log_every': 100,
            'val_every': 1000,
            'checkpoint_every': 5000,
            'convergence_test_every': 5000,
            'grad_clip': 1.0
        },
        
        'use_amp': True  # Automatic mixed precision
    }
    
    # Get experiment-specific config
    config = get_experiment_config(args.experiment, base_config)
    
    # Log configuration
    logger.info("="*80)
    logger.info(f"Running experiment: {config['experiment_name']}")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    logger.info("="*80)
    
    # Create trainer and run
    trainer = AEGUDTrainer(config, device=args.device)
    results = trainer.train()
    
    # Print final results
    logger.info("="*80)
    logger.info("Training completed!")
    logger.info(f"Final results: {json.dumps(results, indent=2)}")
    logger.info("="*80)


if __name__ == "__main__":
    main()