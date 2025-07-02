#!/usr/bin/env python3
"""
Simplified AEGUD experiment runner for PoC
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model import SEDD
from data import get_dataloaders
from noise_lib import GeometricNoise, LogLinearNoise  
from graph_lib import Uniform, Absorbing
from model.ema import ExponentialMovingAverage

# Import AEGUD modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aegud.src.entropy_estimator import EntropyEstimator, AdaptiveTransitionMatrix
from aegud.src.adaptive_uniform_graph import AdaptiveUniform
from aegud.src.information_preserving_noise import InformationPreservingNoise
from aegud.src.adaptive_losses import adaptive_score_entropy_loss
from aegud.src.simple_loss import score_entropy_loss


class SimpleExperimentRunner:
    def __init__(self, device="cuda:1"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results_dir = Path("experiments/aegud/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using device: {self.device}")
        
    def create_model(self, vocab_size, model_size="small", graph_type="uniform"):
        """Create SEDD model."""
        if model_size == "small":
            config = {
                "tokens": vocab_size,
                "graph": {
                    "type": "absorb" if graph_type == "absorb" else "uniform"
                },
                "model": {
                    "hidden_size": 768,
                    "n_heads": 12,
                    "n_blocks": 12,
                    "cond_dim": 128,
                    "dropout": 0.0,
                    "scale_by_sigma": graph_type == "absorb"
                }
            }
        else:
            raise ValueError(f"Unknown model size: {model_size}")
            
        model = SEDD(config).to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        return model
        
    def run_experiment(self, exp_name, graph_type, noise_type, num_steps=1000):
        """Run a single experiment."""
        print(f"\n{'='*60}")
        print(f"Running Experiment: {exp_name}")
        print(f"Graph: {graph_type}, Noise: {noise_type}")
        print(f"{'='*60}")
        
        # Configuration
        vocab_size = 50257  # GPT-2 vocab size
        batch_size = 16
        seq_len = 128
        lr = 3e-4
        
        # Create components
        model = self.create_model(vocab_size, graph_type=graph_type)
        ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
        
        # Create graph
        if graph_type == "uniform":
            graph = Uniform(vocab_size)
        elif graph_type == "absorb":
            graph = Absorbing(vocab_size)
        elif graph_type == "adaptive_uniform":
            entropy_estimator = EntropyEstimator(vocab_size).to(self.device)
            transition_matrix = AdaptiveTransitionMatrix(vocab_size).to(self.device)
            graph = AdaptiveUniform(
                vocab_size,
                entropy_estimator=entropy_estimator,
                transition_matrix=transition_matrix,
                entropy_scale=1.0,
                sparsity_k=100
            )
            # Move graph components to device
            graph = graph.to(self.device)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
            
        # Create noise schedule
        if noise_type == "geometric":
            noise = GeometricNoise()
        elif noise_type == "loglinear":
            noise = LogLinearNoise()
        elif noise_type == "information_preserving":
            if hasattr(graph, 'entropy_estimator'):
                noise = InformationPreservingNoise(
                    entropy_estimator=graph.entropy_estimator
                )
            else:
                noise = InformationPreservingNoise()
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
            
        # Optimizer
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + 
            (list(graph.entropy_estimator.parameters()) + 
             list(graph.transition_matrix.parameters()) 
             if graph_type == "adaptive_uniform" else []),
            lr=lr
        )
        
        # Results tracking
        results = {
            "losses": [],
            "eval_losses": [],
            "timestamps": []
        }
        
        # Training loop
        print("\nStarting training...")
        start_time = time.time()
        
        for step in range(num_steps):
            # Generate random data for PoC
            x_0 = torch.randint(0, vocab_size, (batch_size, seq_len)).to(self.device)
            
            # Compute loss
            if graph_type == "adaptive_uniform":
                loss, loss_dict = adaptive_score_entropy_loss(
                    model, x_0, graph, noise,
                    entropy_regularizer=0.01,
                    info_preservation_weight=0.1
                )
            else:
                loss = score_entropy_loss(model, x_0, graph, noise)
                loss_dict = {"total": loss.item()}
                
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model.parameters())
            
            # Log progress
            if step % 50 == 0:
                elapsed = time.time() - start_time
                results["losses"].append(loss.item())
                results["timestamps"].append(elapsed)
                print(f"Step {step}/{num_steps}, Loss: {loss.item():.4f}, "
                      f"Time: {elapsed:.1f}s")
                
                # Quick evaluation
                if step % 200 == 0 and step > 0:
                    eval_loss = self.evaluate(model, ema, graph, noise, vocab_size)
                    results["eval_losses"].append(eval_loss)
                    print(f"  Eval loss: {eval_loss:.4f}")
                    
        # Save results
        results["final_loss"] = results["losses"][-1] if results["losses"] else None
        results["total_time"] = time.time() - start_time
        results["exp_name"] = exp_name
        results["graph_type"] = graph_type
        results["noise_type"] = noise_type
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"{exp_name}_{timestamp}.json"
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"\nExperiment completed in {results['total_time']:.1f}s")
        print(f"Final loss: {results['final_loss']:.4f}")
        print(f"Results saved to: {result_file}")
        
        return results
        
    def evaluate(self, model, ema, graph, noise, vocab_size):
        """Quick evaluation."""
        model.eval()
        with torch.no_grad():
            # Store current parameters and apply EMA
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            
            # Generate random eval data
            x_0 = torch.randint(0, vocab_size, (8, 128)).to(self.device)
            loss = score_entropy_loss(model, x_0, graph, noise)
            
            # Restore original parameters
            ema.restore(model.parameters())
            
        model.train()
        return loss.item()
        
    def run_all_experiments(self):
        """Run all comparison experiments."""
        experiments = [
            ("baseline_uniform", "uniform", "geometric"),
            ("baseline_absorb", "absorb", "loglinear"),
            ("adaptive_uniform", "adaptive_uniform", "information_preserving"),
        ]
        
        all_results = {}
        
        for exp_name, graph_type, noise_type in experiments:
            try:
                results = self.run_experiment(exp_name, graph_type, noise_type, num_steps=500)
                all_results[exp_name] = results
            except Exception as e:
                print(f"Error in experiment {exp_name}: {e}")
                import traceback
                traceback.print_exc()
                
        # Create comparison plot
        self.plot_comparison(all_results)
        
        return all_results
        
    def plot_comparison(self, all_results):
        """Create comparison plots."""
        plt.figure(figsize=(12, 6))
        
        # Plot training losses
        plt.subplot(1, 2, 1)
        for exp_name, results in all_results.items():
            if "losses" in results and results["losses"]:
                steps = np.arange(0, len(results["losses"]) * 50, 50)
                plt.plot(steps, results["losses"], label=exp_name, linewidth=2)
                
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot final losses
        plt.subplot(1, 2, 2)
        exp_names = []
        final_losses = []
        
        for exp_name, results in all_results.items():
            if "final_loss" in results and results["final_loss"] is not None:
                exp_names.append(exp_name.replace("_", "\n"))
                final_losses.append(results["final_loss"])
                
        bars = plt.bar(exp_names, final_losses, color=['blue', 'green', 'red'])
        plt.ylabel("Final Loss")
        plt.title("Final Loss Comparison")
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, loss in zip(bars, final_losses):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{loss:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nComparison plot saved to: {plot_file}")
        
        # Print summary table
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(f"{'Experiment':<20} {'Final Loss':<15} {'Time (s)':<15}")
        print("-"*60)
        
        for exp_name, results in all_results.items():
            final_loss = results.get('final_loss', 'N/A')
            total_time = results.get('total_time', 'N/A')
            
            if isinstance(final_loss, float):
                loss_str = f"{final_loss:.4f}"
            else:
                loss_str = str(final_loss)
                
            if isinstance(total_time, float):
                time_str = f"{total_time:.1f}"
            else:
                time_str = str(total_time)
                
            print(f"{exp_name:<20} {loss_str:<15} {time_str:<15}")
            
        print("="*60)
        
        # Check if adaptive uniform is winning
        if all([exp in all_results for exp in ["baseline_uniform", "baseline_absorb", "adaptive_uniform"]]):
            uniform_loss = all_results["baseline_uniform"].get("final_loss", float('inf'))
            absorb_loss = all_results["baseline_absorb"].get("final_loss", float('inf'))
            adaptive_loss = all_results["adaptive_uniform"].get("final_loss", float('inf'))
            
            print("\nRESULT ANALYSIS:")
            print(f"Baseline Uniform Loss: {uniform_loss:.4f}")
            print(f"Baseline Absorb Loss: {absorb_loss:.4f}")
            print(f"Adaptive Uniform Loss: {adaptive_loss:.4f}")
            
            if adaptive_loss < uniform_loss and adaptive_loss < absorb_loss:
                print("\n🎉 SUCCESS: Adaptive Uniform WINS! 🎉")
                print(f"Improvement over Uniform: {(uniform_loss - adaptive_loss) / uniform_loss * 100:.1f}%")
                print(f"Improvement over Absorb: {(absorb_loss - adaptive_loss) / absorb_loss * 100:.1f}%")
            elif adaptive_loss < uniform_loss:
                print("\n✓ Adaptive Uniform beats baseline Uniform")
                print(f"Improvement: {(uniform_loss - adaptive_loss) / uniform_loss * 100:.1f}%")
                print("Still working on beating Absorb...")
            else:
                print("\n⚠️  Adaptive Uniform needs more tuning...")
                print("Consider adjusting hyperparameters or training longer")


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--experiment", type=str, default="all")
    args = parser.parse_args()
    
    runner = SimpleExperimentRunner(device=args.device)
    
    if args.experiment == "all":
        runner.run_all_experiments()
    else:
        # Run single experiment
        if args.experiment == "baseline_uniform":
            runner.run_experiment("baseline_uniform", "uniform", "geometric")
        elif args.experiment == "baseline_absorb":
            runner.run_experiment("baseline_absorb", "absorb", "loglinear")
        elif args.experiment == "adaptive_uniform":
            runner.run_experiment("adaptive_uniform", "adaptive_uniform", "information_preserving")


if __name__ == "__main__":
    main()