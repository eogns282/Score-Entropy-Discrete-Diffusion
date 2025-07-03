# Enhanced AEGUD (Adaptive Entropy-Guided Uniform Diffusion) Research

This directory contains the implementation and experiments for Enhanced AEGUD, addressing the fundamental challenge of making uniform state transitions competitive with absorbing state in discrete diffusion models.

## Overview

Based on the insights from NEW_NEXT_IDEA.md, we have implemented multiple novel research directions to improve discrete diffusion models, focusing on the trade-off between information preservation and theoretical convergence.

## Key Implementations

### 1. Enhanced AEGUD V2 (`src/enhanced_adaptive_uniform_graph_v2.py`)
The core implementation with multiple enhancement strategies:
- **Vocabulary-Aware Decay**: Token-specific decay rates based on importance
- **Learnable Convergence Schedule**: Neural network learns optimal transition timing
- **Information Bottleneck**: Principled information-theoretic compression
- **Relaxed Convergence**: Accept ε-approximate uniform distributions
- **Fixed Shape Mismatches**: Resolved tensor dimension issues from V1

### 2. Hierarchical Diffusion (`src/hierarchical_diffusion.py`)
Multi-scale diffusion at different semantic levels:
- Level 0: Topic/style tokens (coarse-grained)
- Level 1: Syntactic structure tokens
- Level 2: Word-level tokens (fine-grained)

### 3. Enhanced Metrics (`src/enhanced_metrics.py`)
Comprehensive evaluation framework:
- Semantic preservation metrics (n-gram overlap, position similarity)
- Generation diversity analysis (self-BLEU, token entropy)
- Convergence quality measurement (entropy, KL divergence, χ² tests)
- Scale-dependent analysis capabilities

## Running Experiments

### Quick Test
```bash
# Test the implementation with small parameters
python experiments/aegud/test_scaled_setup.py
```

### Scaled Experiments
```bash
# Run experiments sequentially on single GPU
./experiments/aegud/launch_scaled_experiments.sh sequential

# Run experiments in parallel on multiple GPUs
./experiments/aegud/launch_scaled_experiments.sh parallel

# Run specific experiment with real WikiText data
./experiments/aegud/launch_scaled_experiments.sh real_data enhanced_v2_full
```

### Individual Experiments
```bash
# Run specific experiment
python experiments/aegud/run_final_scaled_experiments.py \
    --experiment enhanced_v2_vocab_aware \
    --device cuda:0 \
    --vocab_size 5000 \
    --batch_size 128 \
    --num_steps 100000 \
    --use_wandb
```

## Available Experiments

1. **baseline_uniform**: Standard uniform graph baseline
2. **original_aegud**: Original AEGUD implementation
3. **enhanced_v2_vocab_aware**: Enhanced AEGUD with vocabulary-aware decay
4. **enhanced_v2_info_bottleneck**: Enhanced AEGUD with information bottleneck
5. **enhanced_v2_full**: Enhanced AEGUD with all features combined

## Key Findings

### Fundamental Trade-off
```
Information Preservation ↔ Proper Convergence
```
This trade-off is inherent to discrete diffusion models, not just an implementation issue.

### Performance Summary
- **Vocabulary-Aware Decay**: Best information preservation
- **Information Bottleneck**: Principled compression with good results
- **Relaxed Convergence**: Practical solution to theoretical limitations
- **Full Enhanced**: Most complex but potentially best overall

### Convergence Reality
- Perfect uniform convergence (KL < 0.01) is extremely difficult
- KL divergence plateaus around 0.26-0.27 for discrete models
- Relaxed criteria (ε-approximate) provides practical path forward

## File Structure

```
experiments/aegud/
├── src/
│   ├── adaptive_uniform_graph.py          # Original AEGUD
│   ├── enhanced_adaptive_uniform_graph.py # Enhanced V1 (with issues)
│   ├── enhanced_adaptive_uniform_graph_v2.py # Fixed & extended V2
│   ├── hierarchical_diffusion.py         # Multi-scale diffusion
│   ├── enhanced_metrics.py               # Comprehensive metrics
│   ├── entropy_estimator.py              # Entropy estimation module
│   ├── information_preserving_noise.py   # Info-preserving schedules
│   ├── simple_loss.py                    # Basic loss function
│   └── diffusion_validator.py            # Validation tools
├── run_final_scaled_experiments.py       # Main experiment runner
├── launch_scaled_experiments.sh          # Batch experiment launcher
├── test_scaled_setup.py                  # Quick test script
├── FINAL_RESULTS_SUMMARY.md             # Comprehensive results
├── NEW_NEXT_IDEA.md                     # Research directions
└── README.md                            # This file
```

## Requirements

- PyTorch 2.0+ with CUDA support
- 8x NVIDIA RTX A6000 GPUs (or similar)
- Python 3.9+
- Additional dependencies in environment.yml

## Future Directions

1. **Theoretical Analysis**
   - Prove convergence bounds for relaxed criteria
   - Connect to optimal transport theory
   - Analyze information-theoretic properties

2. **Practical Applications**
   - Domain-specific implementations (code, proteins, music)
   - Controllable generation with adaptive approaches
   - Efficient sampling algorithms

3. **Novel Extensions**
   - Adaptive vocabulary size during diffusion
   - Semantic graph structures using knowledge graphs
   - Multi-modal discrete diffusion

## Citation

If you use this code in your research, please cite:
```
[Your citation here when published]
```

## Acknowledgments

This research builds upon the original SEDD implementation and extends it with novel approaches for improving discrete diffusion models.