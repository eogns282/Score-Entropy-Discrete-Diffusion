#!/usr/bin/env python3
"""
Run AEGUD experiments: Baseline comparisons and adaptive uniform experiments
"""

import os
import sys
import torch
import numpy as np
import json
import time
from pathlib import Path
import argparse
from datetime import datetime

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import main codebase modules
from train import train
from model.transformer import SEDD
from data import get_dataloaders
from noise_lib import Geometric, LogLinear
from graph_lib import Uniform, Absorbing
from model.ema import ExponentialMovingAverage
import hydra
from omegaconf import DictConfig, OmegaConf

# Import AEGUD modules
from experiments.aegud.src.entropy_estimator import EntropyEstimator, AdaptiveTransitionMatrix
from experiments.aegud.src.adaptive_uniform_graph import AdaptiveUniform, HierarchicalAdaptiveUniform
from experiments.aegud.src.information_preserving_noise import InformationPreservingNoise, ContentAwareNoise
from experiments.aegud.src.adaptive_losses import adaptive_score_entropy_loss


class ExperimentRunner:
    """Manages and runs AEGUD experiments."""
    
    def __init__(self, base_config_path="configs/config.yaml"):
        self.base_config_path = base_config_path
        self.results_dir = Path("experiments/aegud/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_base_config(self):
        """Load base configuration."""
        with hydra.initialize(config_path="../../configs", version_base=None):
            cfg = hydra.compose(config_name="config")
        return cfg
        
    def create_experiment_config(self, experiment_type, **kwargs):
        """Create configuration for specific experiment type."""
        cfg = self.load_base_config()
        
        # Common settings for PoC experiments
        cfg.model = "small"
        cfg.training.batch_size = 32
        cfg.training.total_iters = 1000  # Quick PoC
        cfg.training.eval_every = 100
        cfg.training.sample_every = 100
        cfg.training.save_every = 500
        cfg.ngpus = 1
        
        # Experiment-specific settings
        if experiment_type == "baseline_uniform":
            cfg.graph.type = "uniform"
            cfg.noise.type = "geometric"
            cfg.model.scale_by_sigma = False
            
        elif experiment_type == "baseline_absorb":
            cfg.graph.type = "absorb"
            cfg.noise.type = "loglinear"
            cfg.model.scale_by_sigma = True
            
        elif experiment_type == "adaptive_uniform":
            cfg.graph.type = "adaptive_uniform"
            cfg.noise.type = "information_preserving"
            cfg.model.scale_by_sigma = False
            # Add custom parameters
            cfg.graph.entropy_scale = kwargs.get("entropy_scale", 1.0)
            cfg.graph.sparsity_k = kwargs.get("sparsity_k", None)
            
        elif experiment_type == "hierarchical_adaptive":
            cfg.graph.type = "hierarchical_adaptive"
            cfg.noise.type = "content_aware"
            cfg.model.scale_by_sigma = False
            cfg.graph.num_levels = kwargs.get("num_levels", 3)
            
        # Override with any additional kwargs
        for key, value in kwargs.items():
            OmegaConf.update(cfg, key, value)
            
        return cfg
        
    def initialize_model_and_components(self, cfg, experiment_type):
        """Initialize model, graph, and noise based on experiment type."""
        device = torch.device(f"cuda:{cfg.local_rank}")
        
        # Initialize noise schedule
        if cfg.noise.type == "geometric":
            noise = Geometric()
        elif cfg.noise.type == "loglinear":
            noise = LogLinear()
        elif cfg.noise.type == "information_preserving":
            # Need entropy estimator for adaptive noise
            entropy_estimator = EntropyEstimator(cfg.tokens)
            noise = InformationPreservingNoise(entropy_estimator=entropy_estimator)
        elif cfg.noise.type == "content_aware":
            entropy_estimator = EntropyEstimator(cfg.tokens)
            noise = ContentAwareNoise(entropy_estimator=entropy_estimator)
        else:
            noise = Geometric()  # Default
            
        # Initialize graph
        if cfg.graph.type == "uniform":
            graph = Uniform(cfg.tokens)
        elif cfg.graph.type == "absorb":
            graph = Absorbing(cfg.tokens)
        elif cfg.graph.type == "adaptive_uniform":
            entropy_estimator = EntropyEstimator(cfg.tokens)
            transition_matrix = AdaptiveTransitionMatrix(cfg.tokens)
            graph = AdaptiveUniform(
                cfg.tokens,
                entropy_estimator=entropy_estimator,
                transition_matrix=transition_matrix,
                entropy_scale=cfg.graph.get("entropy_scale", 1.0),
                sparsity_k=cfg.graph.get("sparsity_k", None)
            )
        elif cfg.graph.type == "hierarchical_adaptive":
            graph = HierarchicalAdaptiveUniform(
                cfg.tokens,
                num_levels=cfg.graph.get("num_levels", 3)
            )
        else:
            graph = Uniform(cfg.tokens)  # Default
            
        # Initialize model
        model = SEDD(
            graph.dim,
            n_T=cfg.model.n_T,
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
            n_blocks=cfg.model.n_blocks,
            mlp_ratio=cfg.model.mlp_ratio,
            drop_p=cfg.model.drop_p,
            scale_by_sigma=cfg.model.scale_by_sigma
        ).to(device)
        
        # Initialize EMA
        ema = ExponentialMovingAverage(
            model.parameters(),
            decay=cfg.training.ema
        )
        
        return model, graph, noise, ema
        
    def run_poc_experiment(self, experiment_type, **kwargs):
        """Run a proof-of-concept experiment."""
        print(f"\n{'='*60}")
        print(f"Running PoC Experiment: {experiment_type}")
        print(f"{'='*60}\n")
        
        # Create config
        cfg = self.create_experiment_config(experiment_type, **kwargs)
        
        # Set up results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.results_dir / f"poc_{experiment_type}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(exp_dir / "config.yaml", "w") as f:
            OmegaConf.save(cfg, f)
            
        # Initialize components
        model, graph, noise, ema = self.initialize_model_and_components(cfg, experiment_type)
        
        # Get data loaders
        train_loader, eval_loader = get_dataloaders(
            cfg.data.name,
            cfg.training.batch_size,
            cfg.training.eval_batch_size
        )
        
        # Set up optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay
        )
        
        # Training loop
        device = torch.device(f"cuda:{cfg.local_rank}")
        results = {
            "losses": [],
            "eval_metrics": [],
            "samples": []
        }
        
        print("Starting training...")
        for step in range(cfg.training.total_iters):
            # Get batch
            x_0 = next(iter(train_loader))[0].to(device)
            
            # Compute loss
            if experiment_type in ["adaptive_uniform", "hierarchical_adaptive"]:
                loss, loss_dict = adaptive_score_entropy_loss(
                    model, x_0, graph, noise,
                    entropy_regularizer=0.01,
                    info_preservation_weight=0.1
                )
            else:
                from losses import score_entropy_loss
                loss = score_entropy_loss(model, x_0, graph, noise)
                loss_dict = {"total": loss.item()}
                
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update EMA
            ema.update()
            
            # Log results
            if step % 10 == 0:
                results["losses"].append({
                    "step": step,
                    "loss": loss.item(),
                    "loss_dict": loss_dict
                })
                print(f"Step {step}: Loss = {loss.item():.4f}")
                
            # Evaluation
            if step % cfg.training.eval_every == 0 and step > 0:
                print(f"\nEvaluating at step {step}...")
                eval_metrics = self.evaluate_model(
                    model, ema, graph, noise, eval_loader, device
                )
                results["eval_metrics"].append({
                    "step": step,
                    "metrics": eval_metrics
                })
                print(f"Eval perplexity: {eval_metrics.get('perplexity', 'N/A')}")
                
            # Generate samples
            if step % cfg.training.sample_every == 0 and step > 0:
                print(f"\nGenerating samples at step {step}...")
                samples = self.generate_samples(
                    model, ema, graph, noise, device, num_samples=5
                )
                results["samples"].append({
                    "step": step,
                    "samples": samples
                })
                
        # Save results
        with open(exp_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        # Save model checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": OmegaConf.to_container(cfg),
            "final_loss": results["losses"][-1]["loss"] if results["losses"] else None
        }, exp_dir / "checkpoint.pt")
        
        print(f"\nExperiment completed. Results saved to {exp_dir}")
        
        return results, exp_dir
        
    def evaluate_model(self, model, ema, graph, noise, eval_loader, device):
        """Evaluate model performance."""
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            with ema.average_parameters():
                for i, (x_0, _) in enumerate(eval_loader):
                    if i >= 10:  # Limit evaluation batches for PoC
                        break
                        
                    x_0 = x_0.to(device)
                    
                    # Compute loss
                    from losses import score_entropy_loss
                    loss = score_entropy_loss(model, x_0, graph, noise)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
        avg_loss = total_loss / max(num_batches, 1)
        perplexity = np.exp(avg_loss)
        
        model.train()
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity
        }
        
    def generate_samples(self, model, ema, graph, noise, device, num_samples=5):
        """Generate text samples."""
        model.eval()
        samples = []
        
        with torch.no_grad():
            with ema.average_parameters():
                # Simple generation for PoC
                seq_len = 128
                
                for _ in range(num_samples):
                    # Start from noise
                    if hasattr(graph, 'absorb'):
                        # Absorbing state - start from all masks
                        x_T = torch.full((1, seq_len), graph.dim - 1, device=device)
                    else:
                        # Uniform - start from random tokens
                        x_T = torch.randint(0, graph.dim, (1, seq_len), device=device)
                        
                    # Simple generation loop (abbreviated for PoC)
                    x_t = x_T
                    for t in torch.linspace(1, 0, 50):
                        t_batch = torch.full((1,), t.item(), device=device)
                        sigma = noise.sigma(t_batch)
                        
                        # Get score from model
                        score = model(x_t, sigma)
                        
                        # Simple update (placeholder - full sampling is complex)
                        # In practice, you'd use the sampling.py functions
                        noise_level = torch.randn_like(x_t.float()) * 0.1
                        x_t = x_t  # Placeholder
                        
                    samples.append(x_t.cpu().tolist())
                    
        model.train()
        return samples
        
    def run_comparison_experiments(self):
        """Run all comparison experiments."""
        all_results = {}
        
        # 1. Baseline Uniform
        print("\n" + "="*80)
        print("EXPERIMENT 1: Baseline Uniform")
        print("="*80)
        results_uniform, dir_uniform = self.run_poc_experiment("baseline_uniform")
        all_results["baseline_uniform"] = results_uniform
        
        # 2. Baseline Absorb
        print("\n" + "="*80)
        print("EXPERIMENT 2: Baseline Absorb")
        print("="*80)
        results_absorb, dir_absorb = self.run_poc_experiment("baseline_absorb")
        all_results["baseline_absorb"] = results_absorb
        
        # 3. Adaptive Uniform (AEGUD)
        print("\n" + "="*80)
        print("EXPERIMENT 3: Adaptive Uniform (AEGUD)")
        print("="*80)
        results_adaptive, dir_adaptive = self.run_poc_experiment(
            "adaptive_uniform",
            entropy_scale=1.0,
            sparsity_k=100
        )
        all_results["adaptive_uniform"] = results_adaptive
        
        # 4. Hierarchical Adaptive Uniform
        print("\n" + "="*80)
        print("EXPERIMENT 4: Hierarchical Adaptive Uniform")
        print("="*80)
        results_hierarchical, dir_hierarchical = self.run_poc_experiment(
            "hierarchical_adaptive",
            num_levels=3
        )
        all_results["hierarchical_adaptive"] = results_hierarchical
        
        # Save comparison results
        comparison_file = self.results_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_file, "w") as f:
            json.dump(all_results, f, indent=2)
            
        print(f"\n\nAll experiments completed! Comparison saved to {comparison_file}")
        
        # Print summary
        self.print_comparison_summary(all_results)
        
        return all_results
        
    def print_comparison_summary(self, all_results):
        """Print a summary comparison of all experiments."""
        print("\n" + "="*80)
        print("EXPERIMENT COMPARISON SUMMARY")
        print("="*80)
        
        for exp_name, results in all_results.items():
            if results["losses"]:
                final_loss = results["losses"][-1]["loss"]
            else:
                final_loss = "N/A"
                
            if results["eval_metrics"]:
                final_perplexity = results["eval_metrics"][-1]["metrics"].get("perplexity", "N/A")
            else:
                final_perplexity = "N/A"
                
            print(f"\n{exp_name}:")
            print(f"  Final Loss: {final_loss}")
            print(f"  Final Perplexity: {final_perplexity}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run AEGUD experiments")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "baseline_uniform", "baseline_absorb", 
                                "adaptive_uniform", "hierarchical_adaptive"],
                        help="Which experiment to run")
    parser.add_argument("--gpu", type=int, default=1,
                        help="Which GPU to use (default: 1)")
    
    args = parser.parse_args()
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Create experiment runner
    runner = ExperimentRunner()
    
    if args.experiment == "all":
        # Run all comparison experiments
        runner.run_comparison_experiments()
    else:
        # Run single experiment
        runner.run_poc_experiment(args.experiment)


if __name__ == "__main__":
    main()