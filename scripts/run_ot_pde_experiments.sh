#!/bin/bash
#
# Run OT-FFM experiments for PDE datasets (1D and 2D).
#
# This script runs comprehensive OT experiments on PDE datasets:
#
# 1D PDE DATASETS:
#   - kdv               : Deterministic Korteweg-de Vries equation
#   - stochastic_kdv    : Stochastic KdV equation
#   - ginzburg_landau   : Stochastic Ginzburg-Landau equation
#
# 2D PDE DATASETS:
#   - navier_stokes     : Navier-Stokes equation (64x64 vorticity)
#   - stochastic_ns     : Stochastic Navier-Stokes equation (64x64)
#
# OT METHODS:
#   - gaussian      : Closed-form Gaussian OT (Bures-Wasserstein)
#   - exact         : Exact OT solver (EMD)
#   - sinkhorn      : Entropic regularization (reg: 0.1, 0.5, 1.0)
#
# KERNELS (for Sinkhorn/Exact):
#   - euclidean     : Squared Euclidean distance
#   - rbf           : RBF/Gaussian kernel RKHS distance
#   - signature     : Signature kernel (1D PDEs only - for time series structure)
#
# 1D PDEs: 15 configurations (includes signature kernel)
# 2D PDEs: 11 configurations (no signature kernel for 2D data)
#
# Usage:
#   ./run_ot_pde_experiments.sh kdv              # Run KdV experiments
#   ./run_ot_pde_experiments.sh stochastic_kdv   # Run stochastic KdV experiments
#   ./run_ot_pde_experiments.sh ginzburg_landau  # Run Ginzburg-Landau experiments
#   ./run_ot_pde_experiments.sh navier_stokes    # Run Navier-Stokes experiments
#   ./run_ot_pde_experiments.sh stochastic_ns    # Run stochastic NS experiments
#   ./run_ot_pde_experiments.sh 1d               # Run all 1D PDE experiments
#   ./run_ot_pde_experiments.sh 2d               # Run all 2D PDE experiments
#   ./run_ot_pde_experiments.sh all              # Run all PDE experiments
#
# Environment variables:
#   CUDA_DEVICE  - GPU device to use (default: 0)
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Check for CUDA availability
check_cuda() {
    echo -e "${BLUE}Checking CUDA availability...${NC}"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
    
    if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
        echo -e "${YELLOW}Warning: CUDA not available. Running on CPU.${NC}"
    else
        echo -e "${GREEN}CUDA is available.${NC}"
    fi
}

# Print header
print_header() {
    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo ""
}

# Print section
print_section() {
    echo ""
    echo -e "${YELLOW}--- $1 ---${NC}"
    echo ""
}

# Print 1D configuration summary
print_config_summary_1d() {
    echo -e "${CYAN}OT Configurations for 1D PDEs (15 total):${NC}"
    echo "  Baseline:"
    echo "    - independent"
    echo "  Gaussian OT (closed-form):"
    echo "    - gaussian_ot"
    echo "  Euclidean kernel:"
    echo "    - euclidean_exact"
    echo "    - euclidean_sinkhorn_reg0.1, reg0.5, reg1.0"
    echo "  RBF kernel:"
    echo "    - rbf_exact"
    echo "    - rbf_sinkhorn_reg0.1, reg0.5, reg1.0"
    echo "    - rbf_sinkhorn_barycentric"
    echo "  Signature kernel (captures path structure):"
    echo "    - signature_sinkhorn_reg0.1, reg0.5, reg1.0"
    echo "    - signature_sinkhorn_barycentric"
    echo ""
}

# Print 2D configuration summary
print_config_summary_2d() {
    echo -e "${CYAN}OT Configurations for 2D PDEs (11 total):${NC}"
    echo "  Baseline:"
    echo "    - independent"
    echo "  Gaussian OT (closed-form):"
    echo "    - gaussian_ot"
    echo "  Euclidean kernel:"
    echo "    - euclidean_exact"
    echo "    - euclidean_sinkhorn_reg0.1, reg0.5, reg1.0"
    echo "  RBF kernel:"
    echo "    - rbf_exact"
    echo "    - rbf_sinkhorn_reg0.1, reg0.5, reg1.0"
    echo "    - rbf_sinkhorn_barycentric"
    echo -e "  ${MAGENTA}(No signature kernel for 2D data)${NC}"
    echo ""
}

# Run KdV experiments (1D)
run_kdv() {
    print_header "Running KdV (Korteweg-de Vries) OT-FFM Experiments"
    
    cd "$(dirname "$0")"
    
    echo -e "${BLUE}Output directory: ../outputs/kdv_ot/${NC}"
    echo -e "${BLUE}Dataset:${NC}"
    echo "  - Deterministic KdV equation"
    echo "  - Data shape: (n_samples, 1, 512) - 512 spatial points"
    echo ""
    print_config_summary_1d
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE:-0}
    echo -e "${BLUE}Using CUDA device: $CUDA_VISIBLE_DEVICES${NC}"
    
    # Run the script
    START_TIME=$(date +%s)
    
    mkdir -p ../outputs/kdv_ot
    python kdv_ot.py 2>&1 | tee ../outputs/kdv_ot/experiment_log.txt
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${GREEN}KdV experiments completed in $((ELAPSED / 60)) minutes $((ELAPSED % 60)) seconds${NC}"
}

# Run stochastic KdV experiments (1D)
run_stochastic_kdv() {
    print_header "Running Stochastic KdV OT-FFM Experiments"
    
    cd "$(dirname "$0")"
    
    echo -e "${BLUE}Output directory: ../outputs/stochastic_kdv_ot/${NC}"
    echo -e "${BLUE}Dataset:${NC}"
    echo "  - Stochastic KdV equation"
    echo "  - Data shape: (n_samples, 1, 128) - 128 spatial points"
    echo "  - 1200 trajectories × 101 time steps (snapshot mode)"
    echo ""
    print_config_summary_1d
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE:-0}
    echo -e "${BLUE}Using CUDA device: $CUDA_VISIBLE_DEVICES${NC}"
    
    # Run the script
    START_TIME=$(date +%s)
    
    mkdir -p ../outputs/stochastic_kdv_ot
    python stochastic_kdv_ot.py 2>&1 | tee ../outputs/stochastic_kdv_ot/experiment_log.txt
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${GREEN}Stochastic KdV experiments completed in $((ELAPSED / 60)) minutes $((ELAPSED % 60)) seconds${NC}"
}

# Run Ginzburg-Landau experiments (1D)
run_ginzburg_landau() {
    print_header "Running Ginzburg-Landau OT-FFM Experiments"
    
    cd "$(dirname "$0")"
    
    echo -e "${BLUE}Output directory: ../outputs/ginzburg_landau_ot/${NC}"
    echo -e "${BLUE}Dataset:${NC}"
    echo "  - Stochastic Ginzburg-Landau equation"
    echo "  - Data shape: (n_samples, 1, 129) - 129 spatial points"
    echo "  - 1200 trajectories × 51 time steps (snapshot mode)"
    echo ""
    print_config_summary_1d
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE:-0}
    echo -e "${BLUE}Using CUDA device: $CUDA_VISIBLE_DEVICES${NC}"
    
    # Run the script
    START_TIME=$(date +%s)
    
    mkdir -p ../outputs/ginzburg_landau_ot
    python ginzburg_landau_ot.py 2>&1 | tee ../outputs/ginzburg_landau_ot/experiment_log.txt
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${GREEN}Ginzburg-Landau experiments completed in $((ELAPSED / 60)) minutes $((ELAPSED % 60)) seconds${NC}"
}

# Run Navier-Stokes experiments (2D)
run_navier_stokes() {
    print_header "Running Navier-Stokes OT-FFM Experiments (2D)"
    
    cd "$(dirname "$0")"
    
    echo -e "${BLUE}Output directory: ../outputs/navier_stokes_ot/${NC}"
    echo -e "${BLUE}Dataset:${NC}"
    echo "  - 2D Navier-Stokes equation (vorticity formulation)"
    echo "  - Data shape: (n_samples, 1, 64, 64) - 64×64 spatial grid"
    echo "  - 10 trajectories × 15001 time steps (subsampled)"
    echo "  - Training: 20000 samples, Testing: 5000 samples"
    echo ""
    print_config_summary_2d
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE:-0}
    echo -e "${BLUE}Using CUDA device: $CUDA_VISIBLE_DEVICES${NC}"
    
    # Run the script
    START_TIME=$(date +%s)
    
    mkdir -p ../outputs/navier_stokes_ot
    python navier_stokes_ot.py 2>&1 | tee ../outputs/navier_stokes_ot/experiment_log.txt
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${GREEN}Navier-Stokes experiments completed in $((ELAPSED / 60)) minutes $((ELAPSED % 60)) seconds${NC}"
}

# Run stochastic Navier-Stokes experiments (2D)
run_stochastic_ns() {
    print_header "Running Stochastic Navier-Stokes OT-FFM Experiments (2D)"
    
    cd "$(dirname "$0")"
    
    echo -e "${BLUE}Output directory: ../outputs/stochastic_ns_ot/${NC}"
    echo -e "${BLUE}Dataset:${NC}"
    echo "  - 2D Stochastic Navier-Stokes equation"
    echo "  - Data shape: (n_samples, 1, 64, 64) - 64×64 spatial grid"
    echo "  - 10 trajectories × 15001 time steps (subsampled)"
    echo "  - Training: 20000 samples, Testing: 5000 samples"
    echo ""
    print_config_summary_2d
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE:-0}
    echo -e "${BLUE}Using CUDA device: $CUDA_VISIBLE_DEVICES${NC}"
    
    # Run the script
    START_TIME=$(date +%s)
    
    mkdir -p ../outputs/stochastic_ns_ot
    python stochastic_ns_ot.py 2>&1 | tee ../outputs/stochastic_ns_ot/experiment_log.txt
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${GREEN}Stochastic Navier-Stokes experiments completed in $((ELAPSED / 60)) minutes $((ELAPSED % 60)) seconds${NC}"
}

# Run all 1D PDE experiments
run_1d() {
    print_header "Running All 1D PDE OT-FFM Experiments"
    echo -e "${CYAN}Datasets: KdV, Stochastic KdV, Ginzburg-Landau${NC}"
    echo ""
    
    run_kdv
    run_stochastic_kdv
    run_ginzburg_landau
}

# Run all 2D PDE experiments
run_2d() {
    print_header "Running All 2D PDE OT-FFM Experiments"
    echo -e "${CYAN}Datasets: Navier-Stokes, Stochastic Navier-Stokes${NC}"
    echo ""
    
    run_navier_stokes
    run_stochastic_ns
}

# Main script
main() {
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║         OT-FFM PDE Experiment Runner (Sequential, CUDA)          ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    echo "║  1D PDEs: KdV, Stochastic KdV, Ginzburg-Landau                   ║"
    echo "║  2D PDEs: Navier-Stokes, Stochastic Navier-Stokes                ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    echo "║  OT Methods: gaussian, exact, sinkhorn                           ║"
    echo "║  Kernels: euclidean, rbf, signature (1D only)                    ║"
    echo "║  Regularization (Sinkhorn): 0.1, 0.5, 1.0                        ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    # Check input argument
    if [ $# -eq 0 ]; then
        echo "Usage: $0 <experiment>"
        echo ""
        echo "1D PDE experiments (15 OT configs each):"
        echo "  kdv              - Deterministic Korteweg-de Vries equation"
        echo "  stochastic_kdv   - Stochastic KdV equation"
        echo "  ginzburg_landau  - Stochastic Ginzburg-Landau equation"
        echo ""
        echo "2D PDE experiments (11 OT configs each, no signature kernel):"
        echo "  navier_stokes    - 2D Navier-Stokes (vorticity)"
        echo "  stochastic_ns    - 2D Stochastic Navier-Stokes"
        echo ""
        echo "Batch options:"
        echo "  1d               - Run all 1D PDE experiments"
        echo "  2d               - Run all 2D PDE experiments"
        echo "  all              - Run all PDE experiments"
        echo ""
        echo "Environment variables:"
        echo "  CUDA_DEVICE      - GPU device to use (default: 0)"
        echo ""
        echo "Example:"
        echo "  CUDA_DEVICE=1 $0 navier_stokes"
        echo ""
        exit 1
    fi
    
    EXPERIMENT=$1
    
    # Create output directories
    mkdir -p ../outputs/kdv_ot
    mkdir -p ../outputs/stochastic_kdv_ot
    mkdir -p ../outputs/ginzburg_landau_ot
    mkdir -p ../outputs/navier_stokes_ot
    mkdir -p ../outputs/stochastic_ns_ot
    
    # Setup
    check_cuda
    
    TOTAL_START=$(date +%s)
    
    case $EXPERIMENT in
        kdv)
            run_kdv
            ;;
        stochastic_kdv)
            run_stochastic_kdv
            ;;
        ginzburg_landau)
            run_ginzburg_landau
            ;;
        navier_stokes)
            run_navier_stokes
            ;;
        stochastic_ns)
            run_stochastic_ns
            ;;
        1d)
            run_1d
            ;;
        2d)
            run_2d
            ;;
        all)
            run_1d
            run_2d
            ;;
        *)
            echo -e "${RED}Unknown experiment: $EXPERIMENT${NC}"
            echo "Valid options: kdv, stochastic_kdv, ginzburg_landau, navier_stokes, stochastic_ns, 1d, 2d, all"
            exit 1
            ;;
    esac
    
    TOTAL_END=$(date +%s)
    TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
    
    print_header "PDE Experiments Complete!"
    echo -e "${GREEN}Total time: $((TOTAL_ELAPSED / 60)) minutes $((TOTAL_ELAPSED % 60)) seconds${NC}"
    echo ""
    echo -e "${BLUE}Results saved to:${NC}"
    
    if [ "$EXPERIMENT" == "kdv" ] || [ "$EXPERIMENT" == "1d" ] || [ "$EXPERIMENT" == "all" ]; then
        echo "  - ../outputs/kdv_ot/"
    fi
    if [ "$EXPERIMENT" == "stochastic_kdv" ] || [ "$EXPERIMENT" == "1d" ] || [ "$EXPERIMENT" == "all" ]; then
        echo "  - ../outputs/stochastic_kdv_ot/"
    fi
    if [ "$EXPERIMENT" == "ginzburg_landau" ] || [ "$EXPERIMENT" == "1d" ] || [ "$EXPERIMENT" == "all" ]; then
        echo "  - ../outputs/ginzburg_landau_ot/"
    fi
    if [ "$EXPERIMENT" == "navier_stokes" ] || [ "$EXPERIMENT" == "2d" ] || [ "$EXPERIMENT" == "all" ]; then
        echo "  - ../outputs/navier_stokes_ot/"
    fi
    if [ "$EXPERIMENT" == "stochastic_ns" ] || [ "$EXPERIMENT" == "2d" ] || [ "$EXPERIMENT" == "all" ]; then
        echo "  - ../outputs/stochastic_ns_ot/"
    fi
    
    echo ""
    echo -e "${BLUE}Key output files per config:${NC}"
    echo "  - <config_name>/seed_<N>/samples.pt            # Generated samples"
    echo "  - <config_name>/seed_<N>/model.pt              # Trained model"
    echo "  - <config_name>/seed_<N>/training_metrics.json # Training stats"
    echo "  - <config_name>/seed_<N>/quality_metrics.json  # Generation quality"
    echo ""
    echo -e "${BLUE}Aggregated outputs:${NC}"
    echo "  - experiment_summary.json       # Full results summary"
    echo "  - quality_comparison.pdf        # Generation quality comparison"
    echo "  - training_comparison.pdf       # Training metrics comparison"
    echo "  - samples_comparison.pdf        # Visual sample comparison"
    echo ""
    echo -e "${BLUE}2D-specific outputs (for NS experiments):${NC}"
    echo "  - spectrum_comparison_2d.pdf    # Energy spectrum comparison"
    echo "  - mean_variance_comparison.pdf  # Mean/variance field comparison"
    echo ""
}

# Run main
main "$@"

