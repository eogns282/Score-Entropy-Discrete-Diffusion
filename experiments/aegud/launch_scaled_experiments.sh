#!/bin/bash
# Launch script for scaled Enhanced AEGUD experiments

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use GPUs 0-3, adjust as needed
export OMP_NUM_THREADS=4
export PYTHONPATH="${PYTHONPATH}:/home/daehoon/Score-Entropy-Discrete-Diffusion"

# Experiment settings
VOCAB_SIZE=5000
BATCH_SIZE=128
NUM_STEPS=100000
SEED=42

# Create results directory
RESULTS_DIR="experiments/aegud/results/scaled_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

# Log file
LOG_FILE="$RESULTS_DIR/experiment_log.txt"

echo "Starting Enhanced AEGUD Scaled Experiments" | tee $LOG_FILE
echo "===========================================" | tee -a $LOG_FILE
echo "Configuration:" | tee -a $LOG_FILE
echo "  Vocabulary Size: $VOCAB_SIZE" | tee -a $LOG_FILE
echo "  Batch Size: $BATCH_SIZE" | tee -a $LOG_FILE
echo "  Training Steps: $NUM_STEPS" | tee -a $LOG_FILE
echo "  Results Directory: $RESULTS_DIR" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Function to run experiment
run_experiment() {
    local EXPERIMENT_NAME=$1
    local GPU_ID=$2
    
    echo "Starting experiment: $EXPERIMENT_NAME on GPU $GPU_ID" | tee -a $LOG_FILE
    echo "Time: $(date)" | tee -a $LOG_FILE
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python experiments/aegud/run_final_scaled_experiments.py \
        --experiment $EXPERIMENT_NAME \
        --device cuda:0 \
        --seed $SEED \
        --vocab_size $VOCAB_SIZE \
        --batch_size $BATCH_SIZE \
        --num_steps $NUM_STEPS \
        --use_wandb \
        2>&1 | tee "$RESULTS_DIR/${EXPERIMENT_NAME}_log.txt"
    
    echo "Completed experiment: $EXPERIMENT_NAME" | tee -a $LOG_FILE
    echo "Time: $(date)" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
}

# Option 1: Run experiments sequentially on single GPU
run_sequential() {
    echo "Running experiments sequentially..." | tee -a $LOG_FILE
    
    # Baseline
    run_experiment "baseline_uniform" 0
    
    # Original AEGUD
    run_experiment "original_aegud" 0
    
    # Enhanced V2 variants
    run_experiment "enhanced_v2_vocab_aware" 0
    run_experiment "enhanced_v2_info_bottleneck" 0
    run_experiment "enhanced_v2_full" 0
}

# Option 2: Run experiments in parallel on multiple GPUs
run_parallel() {
    echo "Running experiments in parallel..." | tee -a $LOG_FILE
    
    # Launch experiments on different GPUs
    run_experiment "baseline_uniform" 0 &
    PID1=$!
    
    run_experiment "original_aegud" 1 &
    PID2=$!
    
    run_experiment "enhanced_v2_vocab_aware" 2 &
    PID3=$!
    
    run_experiment "enhanced_v2_info_bottleneck" 3 &
    PID4=$!
    
    # Wait for first batch to complete
    wait $PID1 $PID2 $PID3 $PID4
    
    # Run the most complex experiment last
    run_experiment "enhanced_v2_full" 0
}

# Option 3: Run specific experiment with real data
run_with_real_data() {
    local EXPERIMENT_NAME=$1
    
    echo "Running $EXPERIMENT_NAME with real WikiText data..." | tee -a $LOG_FILE
    
    python experiments/aegud/run_final_scaled_experiments.py \
        --experiment $EXPERIMENT_NAME \
        --device cuda:0 \
        --seed $SEED \
        --vocab_size 50257 \
        --batch_size 64 \
        --num_steps 200000 \
        --use_real_data \
        --use_wandb \
        2>&1 | tee "$RESULTS_DIR/${EXPERIMENT_NAME}_real_data_log.txt"
}

# Parse command line arguments
if [ "$1" = "sequential" ]; then
    run_sequential
elif [ "$1" = "parallel" ]; then
    run_parallel
elif [ "$1" = "real_data" ] && [ -n "$2" ]; then
    run_with_real_data $2
else
    echo "Usage: $0 {sequential|parallel|real_data <experiment_name>}"
    echo ""
    echo "Options:"
    echo "  sequential - Run all experiments one after another on GPU 0"
    echo "  parallel   - Run experiments in parallel on GPUs 0-3"
    echo "  real_data  - Run specific experiment with real WikiText data"
    echo ""
    echo "Available experiments:"
    echo "  - baseline_uniform"
    echo "  - original_aegud"
    echo "  - enhanced_v2_vocab_aware"
    echo "  - enhanced_v2_info_bottleneck"
    echo "  - enhanced_v2_full"
    echo ""
    echo "Example:"
    echo "  $0 sequential"
    echo "  $0 parallel"
    echo "  $0 real_data enhanced_v2_full"
    exit 1
fi

echo "All experiments completed!" | tee -a $LOG_FILE
echo "Results saved to: $RESULTS_DIR" | tee -a $LOG_FILE

# Create summary report
echo "" | tee -a $LOG_FILE
echo "Creating summary report..." | tee -a $LOG_FILE

python - << EOF
import json
import os
from pathlib import Path

results_dir = Path("$RESULTS_DIR")
summary = {}

# Collect results from each experiment
for result_file in results_dir.glob("*/final_results.json"):
    with open(result_file, 'r') as f:
        data = json.load(f)
        exp_name = data['experiment_name']
        summary[exp_name] = {
            'final_train_loss': data.get('final_train_loss', 'N/A'),
            'best_val_loss': data.get('best_val_loss', 'N/A'),
            'training_time_hours': data.get('training_time', 0) / 3600
        }

# Save summary
with open(results_dir / "summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

# Print summary
print("\nExperiment Summary:")
print("="*60)
for exp, metrics in summary.items():
    print(f"\n{exp}:")
    print(f"  Final Train Loss: {metrics['final_train_loss']}")
    print(f"  Best Val Loss: {metrics['best_val_loss']}")
    print(f"  Training Time: {metrics['training_time_hours']:.2f} hours")
EOF

echo "" | tee -a $LOG_FILE
echo "Experiment run completed successfully!" | tee -a $LOG_FILE