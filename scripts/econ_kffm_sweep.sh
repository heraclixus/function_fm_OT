#!/bin/bash
# =============================================================================
# Phase 1: Signature Kernel Hyperparameter Sweep for econ1 dataset
# 
# Goal: Improve k-FFM autocorrelation_mse (currently 178x worse than DDPM)
# 
# Usage:
#   ./econ_kffm_sweep.sh                  # Run all configs sequentially
#   ./econ_kffm_sweep.sh --parallel 4     # Run 4 configs in parallel
#   ./econ_kffm_sweep.sh --config sig_leadlag_order2  # Run single config
#   ./econ_kffm_sweep.sh --dry-run        # Show what would be run
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_ROOT}/configs/econ_signature_sweep.yaml"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/econ_signature_sweep"
DATASET="econ1_population"
EPOCHS=100
N_PARALLEL=1
DRY_RUN=false
SINGLE_CONFIG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            N_PARALLEL="$2"
            shift 2
            ;;
        --config)
            SINGLE_CONFIG="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --parallel N     Run N configs in parallel (default: 1)"
            echo "  --config NAME    Run only this config"
            echo "  --output DIR     Output directory (default: outputs/econ_signature_sweep)"
            echo "  --epochs N       Number of epochs (default: 100)"
            echo "  --dataset NAME   Dataset to use (default: econ1_population)"
            echo "  --dry-run        Show what would be run without executing"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# List of configs to run (from the YAML file)
CONFIGS=(
    # Lead-lag experiments (priority)
    "sig_leadlag_order1"
    "sig_leadlag_order2"
    "sig_leadlag_order3"
    
    # Dyadic order sweep
    "sig_order1"
    "sig_order3"
    
    # Static kernel sigma sweep
    "sig_sigma0.5"
    "sig_sigma2.0"
    "sig_sigma5.0"
    
    # Low regularization
    "sig_reg0.01"
    "sig_reg0.05"
    
    # Combined experiments
    "sig_leadlag_reg0.05"
    "sig_leadlag_order2_sigma2"
    
    # Ablations
    "sig_no_time_aug"
    "sig_no_normalize"
)

# If single config specified, only run that
if [[ -n "$SINGLE_CONFIG" ]]; then
    CONFIGS=("$SINGLE_CONFIG")
fi

# Seeds to run
SEEDS=(1 2 4)

echo "============================================================"
echo "Phase 1: Signature Kernel Hyperparameter Sweep"
echo "============================================================"
echo "Config file: $CONFIG_FILE"
echo "Output dir:  $OUTPUT_DIR"
echo "Dataset:     $DATASET"
echo "Epochs:      $EPOCHS"
echo "Parallel:    $N_PARALLEL"
echo "Configs:     ${#CONFIGS[@]}"
echo "Seeds:       ${SEEDS[*]}"
echo "============================================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create log directory
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

# Function to run a single experiment
run_experiment() {
    local config=$1
    local seed=$2
    local log_file="${LOG_DIR}/${config}_seed${seed}.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: $config (seed=$seed)"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  DRY RUN: python econ_ot.py --config-file $CONFIG_FILE --config $config --dataset $DATASET --seed $seed --epochs $EPOCHS --spath $OUTPUT_DIR"
        return 0
    fi
    
    cd "$SCRIPT_DIR"
    python econ_ot.py \
        --config-file "$CONFIG_FILE" \
        --config "$config" \
        --dataset "$DATASET" \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --spath "$OUTPUT_DIR" \
        > "$log_file" 2>&1
    
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: $config (seed=$seed)"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $config (seed=$seed) - see $log_file"
    fi
    return $exit_code
}

# Export function for parallel execution
export -f run_experiment
export CONFIG_FILE OUTPUT_DIR DATASET EPOCHS LOG_DIR DRY_RUN SCRIPT_DIR

# Build list of all (config, seed) pairs
JOBS=()
for config in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        JOBS+=("$config $seed")
    done
done

echo ""
echo "Total jobs: ${#JOBS[@]}"
echo ""

# Run experiments
if [[ $N_PARALLEL -eq 1 ]]; then
    # Sequential execution
    for job in "${JOBS[@]}"; do
        read -r config seed <<< "$job"
        run_experiment "$config" "$seed" || echo "Warning: $config seed=$seed failed"
    done
else
    # Parallel execution using GNU parallel or xargs
    if command -v parallel &> /dev/null; then
        echo "Using GNU parallel with $N_PARALLEL workers..."
        printf '%s\n' "${JOBS[@]}" | parallel -j "$N_PARALLEL" --colsep ' ' run_experiment {1} {2}
    else
        echo "GNU parallel not found, using xargs..."
        printf '%s\n' "${JOBS[@]}" | xargs -P "$N_PARALLEL" -I {} bash -c 'run_experiment $1 $2' _ {}
    fi
fi

echo ""
echo "============================================================"
echo "Sweep complete!"
echo "Results: $OUTPUT_DIR"
echo "Logs:    $LOG_DIR"
echo "============================================================"

# Generate summary if not dry run
if [[ "$DRY_RUN" != "true" ]]; then
    echo ""
    echo "Generating summary..."
    
    # Create a simple results summary
    SUMMARY_FILE="${OUTPUT_DIR}/sweep_summary.txt"
    echo "# Signature Kernel Sweep Results" > "$SUMMARY_FILE"
    echo "# Generated: $(date)" >> "$SUMMARY_FILE"
    echo "# Dataset: $DATASET" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    
    for config in "${CONFIGS[@]}"; do
        echo "Config: $config" >> "$SUMMARY_FILE"
        for seed in "${SEEDS[@]}"; do
            metrics_file="${OUTPUT_DIR}/${DATASET}/${config}/seed_${seed}/quality_metrics.json"
            if [[ -f "$metrics_file" ]]; then
                # Extract autocorrelation_mse using python
                autocorr=$(python -c "import json; d=json.load(open('$metrics_file')); print(f'{d.get(\"autocorrelation_mse\", \"N/A\"):.2e}')" 2>/dev/null || echo "N/A")
                mean_mse=$(python -c "import json; d=json.load(open('$metrics_file')); print(f'{d.get(\"mean_mse\", \"N/A\"):.2e}')" 2>/dev/null || echo "N/A")
                echo "  seed_$seed: autocorr=$autocorr, mean=$mean_mse" >> "$SUMMARY_FILE"
            else
                echo "  seed_$seed: MISSING" >> "$SUMMARY_FILE"
            fi
        done
        echo "" >> "$SUMMARY_FILE"
    done
    
    cat "$SUMMARY_FILE"
    
    echo ""
    echo "To regenerate comparison plots, run:"
    echo "  python econ_ot.py --load-only --spath $OUTPUT_DIR --dataset $DATASET"
fi

