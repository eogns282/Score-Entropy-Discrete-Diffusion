# CLAUDE.md - Score Entropy Discrete Diffusion Project Guide

## Project Overview

This repository implements **Score Entropy Discrete Diffusion (SEDD)**, a novel approach to discrete diffusion modeling for text generation. The implementation is based on the paper "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution" by Aaron Lou, Chenlin Meng, and Stefano Ermon.

**Key Innovation**: Unlike traditional diffusion models that work with continuous data (like images), SEDD is specifically designed for discrete data like text. It models the ratios between probabilities rather than the probabilities directly, which provides better stability for discrete domains.

## Architecture & Core Components

### 1. Model Architecture (`model/`)
- **Main Model**: `SEDD` class in `model/transformer.py`
  - Based on DiT (Diffusion Transformer) architecture
  - Uses adaptive layer normalization (AdaLN) for timestep conditioning
  - Implements rotary positional embeddings (`model/rotary.py`)
  - Supports both small (85M params) and medium (343M params) configurations

### 2. Diffusion Process Components

#### Noise Schedules (`noise_lib.py`)
- **Geometric**: Exponential interpolation between σ_min and σ_max
- **LogLinear**: Specifically designed for absorbing state diffusion

#### Graph Types (`graph_lib.py`)
- **Uniform**: Tokens can transition to any other token with equal probability
- **Absorbing**: Tokens can only transition to a special absorbing state (mask token)

#### Loss Functions (`losses.py`)
- Implements score entropy loss
- Measures model's ability to predict the score (gradient of log probability)

### 3. Training Pipeline

#### Main Training Scripts
- `train.py`: Core training logic with distributed training support
- `run_train.py`: Entry point for training experiments
- Uses Hydra for configuration management
- Supports mixed precision training and gradient accumulation
- Includes EMA (Exponential Moving Average) for stable sampling

#### Key Training Features
- Distributed training via PyTorch DDP
- Automatic checkpointing and resumption
- WandB integration for experiment tracking
- Perplexity evaluation on WikiText-103

### 4. Sampling Pipeline

#### Sampling Scripts
- `sampling.py`: Core sampling algorithms (Euler, Analytic)
- `run_sample.py`: Unconditional text generation
- `run_sample_cond.py`: Conditional text generation with prefix/suffix
- `catsample.py`: Concatenates multiple samples for evaluation

#### Sampling Strategies
- **Euler**: Numerical ODE solver approach
- **Analytic**: Closed-form solution for certain noise schedules
- Supports predictor-corrector methods
- Optional final denoising step

## Configuration Structure

### Hydra Configuration (`configs/`)
```yaml
config.yaml          # Main configuration file
model/
  small.yaml        # Small model (85M params)
  medium.yaml       # Medium model (343M params)
```

### Key Configuration Parameters
- `noise.type`: geometric or loglinear
- `graph.type`: uniform or absorb
- `model.scale_by_sigma`: False for uniform graph
- `training.accum`: Gradient accumulation steps (1 for small, 2 for medium)
- `ngpus`: Number of GPUs for distributed training

## Environment & Dependencies

### Main Dependencies
- PyTorch 2.0.1 with CUDA 11.8
- Flash Attention 2.2.2 (for efficient transformer computation)
- Transformers library (for tokenization and evaluation)
- Hydra (configuration management)
- WandB (experiment tracking)

### Environment Setup
```bash
conda env create -f environment.yml
conda activate sedd
```

## Pretrained Models

Models are available on HuggingFace:
- Small model: `louaaron/sedd-small`
- Medium model: `louaaron/sedd-medium`

Load using:
```python
from load_model import load_model
model, graph, noise = load_model("louaaron/sedd-small")
```

## Common Commands

### Training
```bash
# Train small model with absorbing graph
python run_train.py noise.type=loglinear graph.type=absorb model=small

# Train medium model with uniform graph
python run_train.py noise.type=geometric graph.type=uniform model=medium model.scale_by_sigma=False training.accum=2
```

### Sampling
```bash
# Unconditional generation
python run_sample.py --model_path louaaron/sedd-small --steps 256

# Conditional generation
python run_sample_cond.py --model_path louaaron/sedd-medium --steps 256 --prefix "Once upon a time" --suffix "happily ever after."
```

### Testing & Validation
```bash
# Run linting (if configured)
# python -m flake8 .

# Run type checking (if configured)
# python -m mypy .

# Run tests
python test.py
```

## Project Structure
```
Score-Entropy-Discrete-Diffusion/
├── model/                  # Model architecture
│   ├── transformer.py     # Main SEDD model
│   ├── rotary.py         # Rotary embeddings
│   └── ema.py            # Exponential moving average
├── configs/               # Hydra configuration files
├── noise_lib.py          # Noise schedule implementations
├── graph_lib.py          # Graph type implementations
├── losses.py             # Loss functions
├── sampling.py           # Sampling algorithms
├── train.py              # Training logic
├── data.py               # Dataset handling
└── utils.py              # Utility functions
```

## Key Insights for Development

1. **Discrete vs Continuous**: This model works with discrete tokens, not continuous values. The diffusion process corrupts text by changing tokens according to transition probabilities.

2. **Score Matching**: The model learns to predict scores (gradients of log probabilities) rather than probabilities directly, which is more stable for discrete data.

3. **Graph Choice**: 
   - Use absorbing graph for mask-based generation (similar to BERT-style masking)
   - Use uniform graph for more traditional diffusion behavior

4. **Noise Schedule**: 
   - LogLinear is recommended for absorbing graphs
   - Geometric works well with uniform graphs

5. **Memory Considerations**: The medium model requires significant GPU memory. Use gradient accumulation if needed.

## Troubleshooting

1. **CUDA Version Mismatch**: Ensure torch and flash-attn use the same CUDA version
2. **OOM Errors**: Reduce batch size or increase gradient accumulation steps
3. **Slow Training**: Check if flash attention is properly installed
4. **Sampling Issues**: Ensure the correct graph type and noise schedule match the trained model

## How to Understand This Repository

### Step-by-Step Learning Path

1. **Start with the Theory**
   - Read the [original paper](https://arxiv.org/abs/2310.16834) to understand the theoretical foundation
   - Key concept: SEDD models p(x_{t-1}|x_t) by estimating score functions (ratios of probabilities)
   - Understand why discrete diffusion is different from continuous diffusion

2. **Understand the Core Components**
   - **Begin with `graph_lib.py`**: Understand how tokens transition during forward diffusion
     - Uniform: Any token → any token (exploration-heavy)
     - Absorbing: Any token → [MASK] token (like BERT)
   - **Study `noise_lib.py`**: Learn how noise schedules control the diffusion speed
     - Geometric: Smooth exponential decay
     - LogLinear: Designed for absorbing states
   - **Examine `losses.py`**: See how the score entropy loss is computed
     - The model predicts log p(x_0|x_t) - log p(x'_0|x_t) for different x_0

3. **Trace Through the Training Process**
   - Start with `run_train.py` → `train.py`
   - Follow a single batch through:
     - Data loading (`data.py`)
     - Forward diffusion (adding noise)
     - Model forward pass (`model/transformer.py`)
     - Loss computation (`losses.py`)
     - Backward pass and optimization

4. **Understand the Sampling Process**
   - Study `sampling.py` for the reverse diffusion algorithms
   - Run `run_sample.py` with `--debug` flag (if available) or add print statements
   - Visualize how text evolves from noise/masks to coherent sentences

### Recommended Reading Order
```
1. README.md
2. graph_lib.py (understand forward process)
3. noise_lib.py (understand scheduling)
4. model/transformer.py (understand architecture)
5. losses.py (understand training objective)
6. train.py (understand training loop)
7. sampling.py (understand generation)
```

## How to Run This Repository

### Quick Start Guide

1. **Environment Setup**
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd Score-Entropy-Discrete-Diffusion
   
   # Create conda environment
   conda env create -f environment.yml
   conda activate sedd
   
   # Verify installation
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Using Pretrained Models**
   ```bash
   # Generate unconditional samples
   python run_sample.py --model_path louaaron/sedd-small --steps 256 --num_samples 10
   
   # Generate with specific prefix
   python run_sample_cond.py --model_path louaaron/sedd-small --steps 256 \
     --prefix "The weather today is" --num_samples 5
   
   # Generate with prefix and suffix (infilling)
   python run_sample_cond.py --model_path louaaron/sedd-medium --steps 256 \
     --prefix "The scientist discovered" --suffix "which changed everything."
   ```

3. **Training Your Own Model**
   ```bash
   # Small model training (1-2 GPUs recommended)
   python run_train.py model=small ngpus=1 training.batch_size=64
   
   # Medium model training (4-8 GPUs recommended)
   python run_train.py model=medium ngpus=4 training.batch_size=32 training.accum=2
   
   # Custom configuration
   python run_train.py \
     model=small \
     noise.type=loglinear \
     graph.type=absorb \
     training.total_iters=100000 \
     training.warmup_iters=5000
   ```

4. **Monitoring Training**
   ```bash
   # Check training logs
   tail -f exp_local/*/logs/train.log
   
   # View samples during training
   ls exp_local/*/samples/
   
   # Monitor with tensorboard (if configured)
   tensorboard --logdir exp_local/
   ```

### Advanced Usage

1. **Custom Datasets**
   - Modify `data.py` to load your dataset
   - Ensure it returns tokenized sequences
   - Update `data.name` in config

2. **Multi-Node Training**
   ```bash
   # On SLURM systems
   python run_train.py -m ngpus=8 +slurm.nodes=4
   ```

3. **Debugging**
   ```bash
   # Run with smaller model for debugging
   python run_train.py model=small training.total_iters=100 training.eval_every=50
   ```

## How to Develop New Research Directions

### 1. Architecture Modifications

**Attention Mechanisms**
```python
# In model/transformer.py, experiment with:
- Different attention patterns (local, sparse, etc.)
- Cross-attention for conditional generation
- Memory-efficient attention variants
```

**Model Scaling**
```python
# Create configs/model/large.yaml
- Scale to 1B+ parameters
- Experiment with depth vs width
- Try mixture of experts (MoE)
```

### 2. Novel Noise Schedules

**Create Custom Noise Schedule**
```python
# In noise_lib.py, add:
class CustomNoise(Noise):
    def sigma(self, t):
        # Your custom schedule
        return your_custom_function(t)
```

**Research Ideas**:
- Adaptive noise schedules based on content
- Learned noise schedules
- Token-specific noise rates

### 3. New Graph Structures

**Implement Domain-Specific Graphs**
```python
# In graph_lib.py, add:
class StructuredGraph(Graph):
    # E.g., for code: respect syntax constraints
    # E.g., for music: follow harmonic rules
```

**Research Ideas**:
- Syntax-aware graphs for code generation
- Hierarchical graphs for structured data
- Dynamic graphs that change during generation

### 4. Alternative Loss Functions

**Experiment with Loss Variants**
```python
# In losses.py, try:
- Weighted score entropy loss
- Auxiliary losses (e.g., perplexity matching)
- Contrastive losses
```

### 5. Sampling Innovations

**New Sampling Algorithms**
```python
# In sampling.py, implement:
- Adaptive step sizes
- Guided sampling with classifiers
- Parallel sampling strategies
```

**Research Ideas**:
- Controllable generation with plug-and-play methods
- Faster sampling with distillation
- Interactive editing capabilities

### 6. Applications to New Domains

**Code Generation**
```python
# Modifications needed:
1. Custom tokenizer for programming languages
2. Syntax-aware graph structure
3. Evaluation metrics (e.g., compilation success)
```

**Protein Sequences**
```python
# Modifications needed:
1. Amino acid vocabulary
2. Biological constraint graphs
3. Folding-aware loss functions
```

**Music Generation**
```python
# Modifications needed:
1. MIDI tokenization
2. Temporal dependency modeling
3. Harmony-aware sampling
```

### 7. Theoretical Research

**Research Questions**:
1. **Convergence Analysis**: When does SEDD converge to the true distribution?
2. **Optimal Transport**: Can we frame SEDD as an optimal transport problem?
3. **Connection to Other Models**: Relationship to autoregressive models, VAEs, etc.

### 8. Practical Improvements

**Efficiency**
- Implement caching for faster sampling
- Distill models for deployment
- Quantization strategies

**Robustness**
- Better handling of rare tokens
- Improved stability for long sequences
- Adversarial training

### Setting Up for Research

1. **Create a New Branch**
   ```bash
   git checkout -b your-research-direction
   ```

2. **Experiment Tracking**
   ```python
   # In configs/config.yaml, add:
   experiment:
     name: "your_experiment_name"
     tags: ["research_area", "modification_type"]
   ```

3. **Quick Iteration Setup**
   ```bash
   # Create a test config for fast experiments
   # configs/model/tiny.yaml
   python run_train.py model=tiny training.total_iters=1000
   ```

4. **Evaluation Pipeline**
   ```python
   # Create custom evaluation scripts
   # eval_your_metric.py
   from load_model import load_model
   model, graph, noise = load_model("your_experiment")
   # Your evaluation code
   ```

### Research Best Practices

1. **Start Small**: Test ideas on small models/datasets first
2. **Ablation Studies**: Isolate the impact of each change
3. **Baseline Comparisons**: Always compare against the original SEDD
4. **Document Everything**: Keep detailed notes in experiment folders
5. **Share Results**: Create clear visualizations and tables

### Useful Research Tools

```python
# Debugging utilities
from utils import set_seed, print_model_size
set_seed(42)  # For reproducibility

# Visualization
import matplotlib.pyplot as plt
# Plot noise schedules, loss curves, etc.

# Analysis
import pandas as pd
# Track metrics across experiments
```

## Novel Research Direction: Making Uniform State Win

### The Challenge: Uniform vs Absorb State

Currently, the absorb state (masking) method achieves better quantitative metrics than uniform state. However, absorb state has a fundamental limitation: it collapses all input sequences to a single point in latent space (all mask tokens), potentially losing rich distributional information. This presents an exciting research opportunity: **How can we make uniform state outperform absorb state?**

### Understanding the Current Limitations

**Why Absorb Wins:**
1. **Simplicity**: Binary decision (mask or not) vs choosing from vocabulary size
2. **Clear Signal**: Mask tokens are unambiguous noise markers
3. **Efficiency**: O(d) transitions vs O(d²) for uniform
4. **Proven Success**: Aligns with BERT-style masking

**Why Uniform Struggles:**
1. **Ambiguity**: Hard to distinguish signal from noise
2. **Learning Difficulty**: Too many possible transitions
3. **Computational Cost**: Quadratic in vocabulary size
4. **Weak Guidance**: No clear "noise" endpoint

### Proposed Novel Approach: Adaptive Entropy-Guided Uniform Diffusion (AEGUD)

**Core Insight**: The uniform state's weakness (treating all transitions equally) can become its strength if we make transitions adaptive based on the information content of the sequence.

#### Key Innovation Components:

1. **Entropy-Aware Transition Matrix**
```python
class AdaptiveUniform(Graph):
    def compute_transition_matrix(self, x_t, t):
        # Compute local entropy for each position
        entropy = self.estimate_local_entropy(x_t)
        
        # High entropy → more uniform (noisy regions)
        # Low entropy → more conservative (coherent regions)
        temperature = 1.0 + entropy * self.entropy_scale
        
        # Adaptive transition probabilities
        P = self.base_uniform_matrix / temperature
        return self.normalize(P)
```

2. **Semantic Coherence Preservation**
```python
def semantic_aware_transitions(self):
    # Learn token embeddings that capture semantic similarity
    # Tokens more likely to transition to semantically similar tokens
    similarity_matrix = self.token_embeddings @ self.token_embeddings.T
    transition_probs = F.softmax(similarity_matrix / self.temp, dim=-1)
    return transition_probs
```

3. **Information-Theoretic Noise Schedule**
```python
class InformationPreservingNoise(Noise):
    def sigma(self, t, x_0):
        # Adjust noise based on information content
        info_content = self.estimate_information(x_0)
        # Preserve more information for high-content regions
        return self.base_sigma(t) * (2.0 - info_content)
```

4. **Hierarchical Uniform Diffusion**
   - **Level 1**: Topic/style tokens (coarse-grained)
   - **Level 2**: Syntactic structure tokens
   - **Level 3**: Word-level tokens (fine-grained)

### Implementation Strategy

1. **Phase 1: Entropy Estimation Module**
```python
# In a new file: adaptive_uniform.py
class EntropyEstimator(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.context_encoder = nn.TransformerEncoder(...)
        self.entropy_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, position):
        context = self.context_encoder(x)
        return torch.sigmoid(self.entropy_head(context[position]))
```

2. **Phase 2: Adaptive Graph Implementation**
```python
# Modify graph_lib.py
class AdaptiveUniform(Uniform):
    def __init__(self, dim, entropy_estimator):
        super().__init__(dim)
        self.entropy_estimator = entropy_estimator
        self.learnable_transitions = nn.Parameter(
            torch.eye(dim) + 0.01 * torch.randn(dim, dim)
        )
    
    def rate(self, i, j, x_t, t):
        # Adaptive rate based on context and entropy
        entropy = self.entropy_estimator(x_t, i)
        base_rate = super().rate(i, j)
        semantic_weight = self.learnable_transitions[i, j]
        return base_rate * entropy * semantic_weight
```

3. **Phase 3: Training Modifications**
```python
# Add to losses.py
def adaptive_score_entropy_loss(model, x_0, graph, noise, entropy_regularizer=0.01):
    # Standard score entropy loss
    loss = score_entropy_loss(model, x_0, graph, noise)
    
    # Add entropy regularization to encourage information preservation
    entropy_loss = compute_entropy_preservation_loss(...)
    
    return loss + entropy_regularizer * entropy_loss
```

### Theoretical Advantages

1. **Information Preservation**: Maintains more information about the original sequence throughout diffusion
2. **Adaptive Complexity**: Simple transitions in high-entropy regions, complex in low-entropy regions
3. **Semantic Coherence**: Preserves semantic relationships during diffusion
4. **Computational Efficiency**: Learn sparse transition matrices that approach O(d log d) complexity

### Experimental Validation

1. **Metrics to Track**:
   - Perplexity comparison vs absorb state
   - Diversity of generated samples
   - Semantic coherence scores
   - Information retention through diffusion process

2. **Ablation Studies**:
   - Entropy guidance vs. fixed uniform
   - Semantic transitions vs. random
   - Hierarchical vs. flat diffusion

3. **Visualization**:
   - Plot information content over diffusion steps
   - Visualize learned transition patterns
   - Compare latent space structure (t-SNE/UMAP)

### Why This Could Win

1. **Best of Both Worlds**: Combines uniform's information preservation with absorb's learning efficiency
2. **Theoretical Foundation**: Grounded in information theory and optimal transport
3. **Practical Benefits**: Could generate more diverse and coherent text
4. **Novel Contribution**: First work to make uniform transitions adaptive and content-aware

### Quick Start Implementation

```bash
# 1. Create new branch
git checkout -b adaptive-uniform-diffusion

# 2. Implement entropy estimator
python implement_entropy_estimator.py

# 3. Test on small model
python run_train.py \
    model=small \
    graph.type=adaptive_uniform \
    graph.entropy_scale=0.5 \
    noise.type=information_preserving \
    training.total_iters=10000

# 4. Compare with baseline
python evaluate_uniform_vs_absorb.py --model exp_local/adaptive_uniform
```

This research direction addresses the fundamental limitation of absorb state (information collapse) while solving uniform state's key weakness (learning difficulty), potentially leading to a new state-of-the-art in discrete diffusion models.

## Future Extensions

- Support for other discrete modalities (e.g., music, code)
- Larger model sizes
- Alternative sampling strategies
- Fine-tuning capabilities
- Integration with other text generation frameworks