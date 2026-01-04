#!/bin/bash
# ============================================================================
# Stochastic Navier-Stokes k-FFM Hyperparameter Sweep Script
# ============================================================================
# Purpose: Improve k-FFM to beat DDPM on spectrum_mse
# Current: Best k-FFM spectrum_mse=49,873 | DDPM=432 (115x worse)
# Target: k-FFM spectrum_mse < 500
#
# Usage:
#   ./stochastic_ns_kffm_sweep.sh                    # Run all configs sequentially
#   ./stochastic_ns_kffm_sweep.sh --parallel 4       # Run with 4 parallel jobs
#   ./stochastic_ns_kffm_sweep.sh --dry-run          # Show what would run
#   ./stochastic_ns_kffm_sweep.sh --config gp_kl0.02 --seed 1
# ============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/../configs/stochastic_ns_sweep.yaml"
PYTHON_SCRIPT="${SCRIPT_DIR}/stochastic_ns_ot.py"
OUTPUT_DIR="${SCRIPT_DIR}/../outputs/stochastic_ns_ot"
SEEDS=(1 2 4)
EPOCHS=100
PARALLEL_JOBS=1
DRY_RUN=false
SINGLE_CONFIG=""
SINGLE_SEED=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --config)
            SINGLE_CONFIG="$2"
            shift 2
            ;;
        --seed)
            SINGLE_SEED="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --parallel N     Run N jobs in parallel (default: 1)"
            echo "  --dry-run        Show what would run without executing"
            echo "  --config NAME    Run only this config"
            echo "  --seed N         Run only this seed (use with --config)"
            echo "  --epochs N       Override number of epochs (default: 100)"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Stochastic NS k-FFM Hyperparameter Sweep${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "${RED}Target: Improve spectrum_mse from ~50,000 to <500${NC}"
echo ""

# Check files exist
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${RED}ERROR: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo -e "${RED}ERROR: Python script not found: $PYTHON_SCRIPT${NC}"
    exit 1
fi

# Get list of configs
cd "$SCRIPT_DIR"
CONFIGS=$(python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    configs = yaml.safe_load(f)
print(' '.join(configs.keys()))
")

echo -e "${YELLOW}Config file: ${CONFIG_FILE}${NC}"
echo -e "${YELLOW}Output directory: ${OUTPUT_DIR}${NC}"
echo -e "${YELLOW}Epochs: ${EPOCHS}${NC}"
echo -e "${YELLOW}Seeds: ${SEEDS[*]}${NC}"
echo -e "${YELLOW}Parallel jobs: ${PARALLEL_JOBS}${NC}"
echo ""

# Count total jobs
CONFIG_COUNT=$(echo $CONFIGS | wc -w | tr -d ' ')
TOTAL_JOBS=$((CONFIG_COUNT * ${#SEEDS[@]}))
echo -e "${GREEN}Total configurations: $CONFIG_COUNT${NC}"
echo -e "${GREEN}Total jobs (configs × seeds): $TOTAL_JOBS${NC}"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}DRY RUN - Commands that would be executed:${NC}"
    echo ""
fi

# Function to run a single experiment
run_experiment() {
    local config=$1
    local seed=$2
    
    local cmd="python $PYTHON_SCRIPT --config-file $CONFIG_FILE --config $config --seed $seed --epochs $EPOCHS"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  $cmd"
    else
        echo -e "${BLUE}[$(date '+%H:%M:%S')] Running: $config (seed=$seed)${NC}"
        
        # Run and capture exit code
        if $cmd; then
            echo -e "${GREEN}[$(date '+%H:%M:%S')] Completed: $config (seed=$seed)${NC}"
        else
            echo -e "${RED}[$(date '+%H:%M:%S')] FAILED: $config (seed=$seed)${NC}"
        fi
    fi
}

# Single config mode
if [[ -n "$SINGLE_CONFIG" ]]; then
    if [[ -n "$SINGLE_SEED" ]]; then
        run_experiment "$SINGLE_CONFIG" "$SINGLE_SEED"
    else
        for seed in "${SEEDS[@]}"; do
            run_experiment "$SINGLE_CONFIG" "$seed"
        done
    fi
    exit 0
fi

# Full sweep mode
if [[ "$PARALLEL_JOBS" -gt 1 && "$DRY_RUN" != "true" ]]; then
    # Parallel execution using GNU parallel or xargs
    echo -e "${YELLOW}Running in parallel with $PARALLEL_JOBS jobs...${NC}"
    
    # Create job list
    JOB_FILE=$(mktemp)
    for config in $CONFIGS; do
        for seed in "${SEEDS[@]}"; do
            echo "$config $seed" >> "$JOB_FILE"
        done
    done
    
    # Check for GNU parallel
    if command -v parallel &> /dev/null; then
        cat "$JOB_FILE" | parallel -j "$PARALLEL_JOBS" --colsep ' ' \
            "echo -e '${BLUE}[$(date '+%H:%M:%S')] Running: {1} (seed={2})${NC}' && \
             python $PYTHON_SCRIPT --config-file $CONFIG_FILE --config {1} --seed {2} --epochs $EPOCHS && \
             echo -e '${GREEN}[$(date '+%H:%M:%S')] Completed: {1} (seed={2})${NC}'"
    else
        echo -e "${YELLOW}GNU parallel not found, using xargs (less efficient)${NC}"
        cat "$JOB_FILE" | xargs -P "$PARALLEL_JOBS" -I {} bash -c '
            config=$(echo {} | cut -d" " -f1)
            seed=$(echo {} | cut -d" " -f2)
            echo "Running: $config (seed=$seed)"
            python '"$PYTHON_SCRIPT"' --config-file '"$CONFIG_FILE"' --config $config --seed $seed --epochs '"$EPOCHS"'
        '
    fi
    
    rm "$JOB_FILE"
else
    # Sequential execution
    JOB_NUM=0
    for config in $CONFIGS; do
        for seed in "${SEEDS[@]}"; do
            JOB_NUM=$((JOB_NUM + 1))
            if [[ "$DRY_RUN" != "true" ]]; then
                echo -e "${BLUE}[Job $JOB_NUM/$TOTAL_JOBS]${NC}"
            fi
            run_experiment "$config" "$seed"
        done
    done
fi

if [[ "$DRY_RUN" != "true" ]]; then
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}Sweep Complete!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo -e "${YELLOW}Results saved to: ${OUTPUT_DIR}${NC}"
    echo ""
    echo "To analyze results:"
    echo "  python stochastic_ns_ot.py --load-only"
    echo ""
    
    # Quick summary of spectrum_mse results
    echo -e "${BLUE}Quick Summary of spectrum_mse (lower is better):${NC}"
    echo "Target to beat: DDPM = 432"
    echo ""
    
    for config in $CONFIGS; do
        # Check if results exist
        if [[ -d "${OUTPUT_DIR}/${config}" ]]; then
            # Try to extract spectrum_mse from quality_metrics.json
            spectrum_mse=$(python -c "
import json
import numpy as np
values = []
for seed in [1, 2, 4]:
    try:
        with open('${OUTPUT_DIR}/${config}/seed_' + str(seed) + '/quality_metrics.json') as f:
            data = json.load(f)
            if 'spectrum_mse' in data and data['spectrum_mse'] is not None:
                values.append(data['spectrum_mse'])
    except: pass
if values:
    print(f'{np.mean(values):.1f}')
else:
    print('N/A')
" 2>/dev/null || echo "N/A")
            
            if [[ "$spectrum_mse" != "N/A" ]]; then
                # Compare to baseline
                better=$(python -c "print('✓ BEATS DDPM!' if $spectrum_mse < 432 else '✗')" 2>/dev/null || echo "?")
                echo "  $config: $spectrum_mse $better"
            fi
        fi
    done
fi

