"""
Run scaled experiments in parallel across multiple GPUs.
Efficiently utilizes all available GPU resources.
"""

import os
import sys
import subprocess
import argparse
import time
from datetime import datetime
import json
from pathlib import Path
import multiprocessing as mp

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def run_single_experiment(gpu_id, experiment_name, vocab_size, num_steps, batch_size):
    """Run a single experiment on a specific GPU."""
    
    cmd = [
        sys.executable,
        "experiments/aegud/run_scaled_experiments.py",
        f"--device=cuda:{gpu_id}",
        f"--vocab_size={vocab_size}",
        f"--num_steps={num_steps}",
        f"--batch_size={batch_size}",
        f"--experiment={experiment_name}"
    ]
    
    print(f"[GPU {gpu_id}] Starting experiment: {experiment_name}")
    print(f"[GPU {gpu_id}] Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        # Set CUDA_VISIBLE_DEVICES to ensure process only sees its assigned GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Run the experiment
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[GPU {gpu_id}] Completed {experiment_name} in {elapsed/60:.2f} minutes")
            return {
                'experiment': experiment_name,
                'gpu_id': gpu_id,
                'status': 'success',
                'elapsed_time': elapsed,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            print(f"[GPU {gpu_id}] Failed {experiment_name} after {elapsed/60:.2f} minutes")
            print(f"[GPU {gpu_id}] Error: {result.stderr}")
            return {
                'experiment': experiment_name,
                'gpu_id': gpu_id,
                'status': 'failed',
                'elapsed_time': elapsed,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
    except Exception as e:
        print(f"[GPU {gpu_id}] Exception in {experiment_name}: {str(e)}")
        return {
            'experiment': experiment_name,
            'gpu_id': gpu_id,
            'status': 'error',
            'error': str(e),
            'elapsed_time': time.time() - start_time
        }


def run_parallel_experiments(experiments, num_gpus, vocab_size, num_steps, batch_size):
    """Run experiments in parallel across available GPUs."""
    
    print(f"Running {len(experiments)} experiments across {num_gpus} GPUs")
    print(f"Experiments: {experiments}")
    
    # Create a process pool
    with mp.Pool(processes=num_gpus) as pool:
        # Prepare tasks
        tasks = []
        for i, exp_name in enumerate(experiments):
            gpu_id = i % num_gpus
            tasks.append((gpu_id, exp_name, vocab_size, num_steps, batch_size))
        
        # Run experiments in parallel
        results = pool.starmap(run_single_experiment, tasks)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run scaled Enhanced AEGUD experiments in parallel')
    parser.add_argument('--num_gpus', type=int, default=4,
                       help='Number of GPUs to use')
    parser.add_argument('--vocab_size', type=int, default=5000,
                       help='Vocabulary size for experiments')
    parser.add_argument('--num_steps', type=int, default=20000,
                       help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size per GPU')
    parser.add_argument('--experiments', type=str, nargs='+',
                       default=['Original_AEGUD_Scaled', 'Enhanced_AEGUD_Asymptotic',
                               'Enhanced_AEGUD_TwoStage', 'Enhanced_AEGUD_Full'],
                       help='List of experiments to run')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PARALLEL SCALED EXPERIMENTS")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Number of GPUs: {args.num_gpus}")
    print(f"  - Vocabulary size: {args.vocab_size}")
    print(f"  - Training steps: {args.num_steps}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Total experiments: {len(args.experiments)}")
    print()
    
    # Check available GPUs
    import torch
    available_gpus = torch.cuda.device_count()
    if available_gpus < args.num_gpus:
        print(f"Warning: Requested {args.num_gpus} GPUs but only {available_gpus} available")
        args.num_gpus = available_gpus
    
    # Create results directory
    results_dir = Path("experiments/aegud/parallel_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    start_time = time.time()
    results = run_parallel_experiments(
        args.experiments,
        args.num_gpus,
        args.vocab_size,
        args.num_steps,
        args.batch_size
    )
    
    total_time = time.time() - start_time
    
    # Process results
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    print("\n" + "="*80)
    print("PARALLEL EXECUTION SUMMARY")
    print("="*80)
    print(f"Total execution time: {total_time/60:.2f} minutes")
    print(f"Successful experiments: {len(successful)}/{len(results)}")
    
    if failed:
        print(f"\nFailed experiments:")
        for r in failed:
            print(f"  - {r['experiment']}: {r.get('error', 'Unknown error')}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'configuration': vars(args),
        'total_time': total_time,
        'results': results,
        'summary': {
            'total': len(results),
            'successful': len(successful),
            'failed': len(failed)
        }
    }
    
    summary_path = results_dir / f"parallel_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {summary_path}")
    
    # Print individual experiment results if available
    print("\nExperiment Results:")
    for exp in args.experiments:
        result_files = list(Path("experiments/aegud/scaled_results").glob(f"{exp}_*/results.json"))
        if result_files:
            latest = max(result_files, key=lambda p: p.stat().st_mtime)
            with open(latest, 'r') as f:
                data = json.load(f)
                val = data.get('final_validation', {})
                conv = val.get('convergence', {}).get('convergence', {})
                
                print(f"\n{exp}:")
                print(f"  - Best Val Loss: {data.get('best_val_loss', 'N/A')}")
                if isinstance(data.get('best_val_loss'), (int, float)):
                    print(f"    ({data['best_val_loss']:.4f})")
                print(f"  - Convergence: {'PASSED' if conv.get('converged', False) else 'FAILED'}")
                print(f"  - Final Entropy: {conv.get('entropy_ratio', 'N/A')}")
                print(f"  - Training Time: {data.get('training_time', 0)/3600:.2f} hours")


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()