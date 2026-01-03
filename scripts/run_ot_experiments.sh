#!/bin/bash
#
# Run OT-FFM experiments sequentially on CUDA.
#
# This script runs comprehensive OT experiments with the following configurations:
#
# OT METHODS:
#   - gaussian      : Closed-form Gaussian OT (Bures-Wasserstein)
#   - exact         : Exact OT solver (EMD)
#   - sinkhorn      : Entropic regularization (reg: 0.1, 0.5, 1.0)
#
# Note: unbalanced and partial OT are not included because they don't preserve
# the marginal constraints required for flow matching.
#
# KERNELS (for Sinkhorn/Exact):
#   - euclidean     : Squared Euclidean distance
#   - rbf           : RBF/Gaussian kernel RKHS distance
#   - signature     : Signature kernel (for time series)
#
# COUPLING STRATEGIES:
#   - sample        : Stochastic coupling (sample from OT plan)
#   - barycentric   : Deterministic coupling (barycentric projection)
#
# Full list of configurations per experiment (15 total):
#   1. independent (baseline - no OT)
#   2. gaussian_ot (closed-form Bures-Wasserstein)
#   3. euclidean_exact
#   4. euclidean_sinkhorn_reg0.1
#   5. euclidean_sinkhorn_reg0.5
#   6. euclidean_sinkhorn_reg1.0
#   7. rbf_exact
#   8. rbf_sinkhorn_reg0.1
#   9. rbf_sinkhorn_reg0.5
#   10. rbf_sinkhorn_reg1.0
#   11. rbf_sinkhorn_barycentric
#   12. signature_sinkhorn_reg0.1
#   13. signature_sinkhorn_reg0.5
#   14. signature_sinkhorn_reg1.0
#   15. signature_sinkhorn_barycentric
#
# Usage:
#   ./run_ot_experiments.sh econ       # Run economics experiments
#   ./run_ot_experiments.sh moGP       # Run mixture of GPs experiments
#   ./run_ot_experiments.sh expr_genes # Run gene expression experiments
#   ./run_ot_experiments.sh AEMET      # Run AEMET weather experiments
#   ./run_ot_experiments.sh rBergomi   # Run rBergomi rough volatility experiments
#   ./run_ot_experiments.sh Heston     # Run Heston stochastic volatility experiments
#   ./run_ot_experiments.sh all        # Run all experiments
#
# Environment variables:
#   CUDA_DEVICE  - GPU device to use (default: 0)
#

set -e  # Exit on error

# Colors for ouatput
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

# Print configuration summary
print_config_summary() {
    echo -e "${CYAN}OT Configurations (15 total):${NC}"
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
    echo "  Signature kernel (time series):"
    echo "    - signature_sinkhorn_reg0.1, reg0.5, reg1.0"
    echo "    - signature_sinkhorn_barycentric"
    echo ""
}

# Run moGP experiments
run_moGP() {
    print_header "Running Mixture of GPs OT-FFM Experiments"
    
    cd "$(dirname "$0")"
    
    echo -e "${BLUE}Output directory: ../outputs/moGP_ot_comprehensive/${NC}"
    print_config_summary
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE:-0}
    echo -e "${BLUE}Using CUDA device: $CUDA_VISIBLE_DEVICES${NC}"
    
    # Run the script
    START_TIME=$(date +%s)
    
    python moGP_ot.py 2>&1 | tee ../outputs/moGP_ot_comprehensive/experiment_log.txt
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${GREEN}moGP experiments completed in $((ELAPSED / 60)) minutes $((ELAPSED % 60)) seconds${NC}"
}

# Run economics experiments
run_econ() {
    print_header "Running Economics Time Series OT-FFM Experiments"
    
    cd "$(dirname "$0")"
    
    echo -e "${BLUE}Output directory: ../outputs/econ_ot_comprehensive/${NC}"
    echo -e "${BLUE}Datasets:${NC}"
    echo "  - econ1_population"
    echo "  - econ2_gdp"
    echo "  - econ3_labor"
    echo ""
    print_config_summary
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE:-0}
    echo -e "${BLUE}Using CUDA device: $CUDA_VISIBLE_DEVICES${NC}"
    
    # Run the script
    START_TIME=$(date +%s)
    
    python econ_ot.py 2>&1 | tee ../outputs/econ_ot_comprehensive/experiment_log.txt
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${GREEN}Economics experiments completed in $((ELAPSED / 60)) minutes $((ELAPSED % 60)) seconds${NC}"
}

# Run gene expression experiments
run_expr_genes() {
    print_header "Running Gene Expression OT-FFM Experiments"
    
    cd "$(dirname "$0")"
    
    echo -e "${BLUE}Output directory: ../outputs/expr_genes_ot_comprehensive/${NC}"
    echo -e "${BLUE}Dataset:${NC}"
    echo "  - Gene expression time series"
    echo ""
    print_config_summary
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE:-0}
    echo -e "${BLUE}Using CUDA device: $CUDA_VISIBLE_DEVICES${NC}"
    
    # Run the script
    START_TIME=$(date +%s)
    
    mkdir -p ../outputs/expr_genes_ot_comprehensive
    python expr_genes_ot.py 2>&1 | tee ../outputs/expr_genes_ot_comprehensive/experiment_log.txt
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${GREEN}Gene expression experiments completed in $((ELAPSED / 60)) minutes $((ELAPSED % 60)) seconds${NC}"
}

# Run AEMET weather experiments
run_AEMET() {
    print_header "Running AEMET Weather OT-FFM Experiments"
    
    cd "$(dirname "$0")"
    
    echo -e "${BLUE}Output directory: ../outputs/AEMET_ot_comprehensive/${NC}"
    echo -e "${BLUE}Dataset:${NC}"
    echo "  - Spanish weather station temperature data (365 days)"
    echo ""
    print_config_summary
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE:-0}
    echo -e "${BLUE}Using CUDA device: $CUDA_VISIBLE_DEVICES${NC}"
    
    # Run the script
    START_TIME=$(date +%s)
    
    mkdir -p ../outputs/AEMET_ot_comprehensive
    python AEMET_ot.py 2>&1 | tee ../outputs/AEMET_ot_comprehensive/experiment_log.txt
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${GREEN}AEMET weather experiments completed in $((ELAPSED / 60)) minutes $((ELAPSED % 60)) seconds${NC}"
}

# Run rBergomi rough volatility experiments
run_rBergomi() {
    print_header "Running rBergomi Rough Volatility OT-FFM Experiments"
    
    cd "$(dirname "$0")"
    
    echo -e "${BLUE}Output directory: ../outputs/rBergomi_ot_H0p10/${NC}"
    echo -e "${BLUE}Dataset:${NC}"
    echo "  - rBergomi log-variance paths (H=0.1, very rough)"
    echo ""
    print_config_summary
    echo -e "${CYAN}Note: Signature kernel should excel on rough paths!${NC}"
    echo ""
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE:-0}
    echo -e "${BLUE}Using CUDA device: $CUDA_VISIBLE_DEVICES${NC}"
    
    # Run the script
    START_TIME=$(date +%s)
    
    mkdir -p ../outputs/rBergomi_ot_H0p10
    python rBergomi_ot.py 2>&1 | tee ../outputs/rBergomi_ot_H0p10/experiment_log.txt
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${GREEN}rBergomi experiments completed in $((ELAPSED / 60)) minutes $((ELAPSED % 60)) seconds${NC}"
}

# Run Heston stochastic volatility experiments
run_Heston() {
    print_header "Running Heston Stochastic Volatility OT-FFM Experiments"
    
    cd "$(dirname "$0")"
    
    echo -e "${BLUE}Output directory: ../outputs/Heston_ot_kappa1.0/${NC}"
    echo -e "${BLUE}Dataset:${NC}"
    echo "  - Heston log-variance paths (smoother baseline)"
    echo ""
    print_config_summary
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE:-0}
    echo -e "${BLUE}Using CUDA device: $CUDA_VISIBLE_DEVICES${NC}"
    
    # Run the script
    START_TIME=$(date +%s)
    
    mkdir -p ../outputs/Heston_ot_kappa1.0
    python Heston_ot.py 2>&1 | tee ../outputs/Heston_ot_kappa1.0/experiment_log.txt
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${GREEN}Heston experiments completed in $((ELAPSED / 60)) minutes $((ELAPSED % 60)) seconds${NC}"
}

# Main script
main() {
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║     Comprehensive OT-FFM Experiment Runner (Sequential, CUDA)    ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    echo "║  OT Methods: gaussian, exact, sinkhorn                            ║"
    echo "║  Kernels: euclidean, rbf, signature                              ║"
    echo "║  Regularization (Sinkhorn): 0.1, 0.5, 1.0                         ║"
    echo "║  Sinkhorn max iterations: 1500                                    ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    # Check input argument
    if [ $# -eq 0 ]; then
        echo "Usage: $0 <experiment>"
        echo ""
        echo "Options:"
        echo "  econ       - Run economics time series experiments"
        echo "  moGP       - Run mixture of GPs experiments"
        echo "  expr_genes - Run gene expression experiments"
        echo "  AEMET      - Run AEMET weather experiments"
        echo "  rBergomi   - Run rBergomi rough volatility experiments"
        echo "  Heston     - Run Heston stochastic volatility experiments"
        echo "  stochvol   - Run both rBergomi and Heston (volatility comparison)"
        echo "  all        - Run all experiments"
        echo ""
        echo "Environment variables:"
        echo "  CUDA_DEVICE  - GPU device to use (default: 0)"
        echo ""
        echo "Example:"
        echo "  CUDA_DEVICE=1 $0 rBergomi"
        echo ""
        echo "Each experiment runs 15 OT configurations:"
        print_config_summary
        exit 1
    fi
    
    EXPERIMENT=$1
    
    # Create output directories
    mkdir -p ../outputs/moGP_ot_comprehensive
    mkdir -p ../outputs/econ_ot_comprehensive
    mkdir -p ../outputs/expr_genes_ot_comprehensive
    mkdir -p ../outputs/AEMET_ot_comprehensive
    mkdir -p ../outputs/rBergomi_ot_H0p10
    mkdir -p ../outputs/Heston_ot_kappa1.0
    
    # Setup
    check_cuda
    
    TOTAL_START=$(date +%s)
    
    case $EXPERIMENT in
        econ)
            run_econ
            ;;
        moGP)
            run_moGP
            ;;
        expr_genes)
            run_expr_genes
            ;;
        AEMET)
            run_AEMET
            ;;
        rBergomi)
            run_rBergomi
            ;;
        Heston)
            run_Heston
            ;;
        stochvol)
            run_rBergomi
            run_Heston
            ;;
        all)
            run_moGP
            run_econ
            run_expr_genes
            run_AEMET
            run_rBergomi
            run_Heston
            ;;
        *)
            echo -e "${RED}Unknown experiment: $EXPERIMENT${NC}"
            echo "Valid options: econ, moGP, expr_genes, AEMET, rBergomi, Heston, stochvol, all"
            exit 1
            ;;
    esac
    
    TOTAL_END=$(date +%s)
    TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
    
    print_header "All Experiments Complete!"
    echo -e "${GREEN}Total time: $((TOTAL_ELAPSED / 60)) minutes $((TOTAL_ELAPSED % 60)) seconds${NC}"
    echo ""
    echo -e "${BLUE}Results saved to:${NC}"
    
    if [ "$EXPERIMENT" == "moGP" ] || [ "$EXPERIMENT" == "all" ]; then
        echo "  - ../outputs/moGP_ot_comprehensive/"
    fi
    if [ "$EXPERIMENT" == "econ" ] || [ "$EXPERIMENT" == "all" ]; then
        echo "  - ../outputs/econ_ot_comprehensive/"
    fi
    if [ "$EXPERIMENT" == "expr_genes" ] || [ "$EXPERIMENT" == "all" ]; then
        echo "  - ../outputs/expr_genes_ot_comprehensive/"
    fi
    if [ "$EXPERIMENT" == "AEMET" ] || [ "$EXPERIMENT" == "all" ]; then
        echo "  - ../outputs/AEMET_ot_comprehensive/"
    fi
    if [ "$EXPERIMENT" == "rBergomi" ] || [ "$EXPERIMENT" == "stochvol" ] || [ "$EXPERIMENT" == "all" ]; then
        echo "  - ../outputs/rBergomi_ot_H0p10/"
    fi
    if [ "$EXPERIMENT" == "Heston" ] || [ "$EXPERIMENT" == "stochvol" ] || [ "$EXPERIMENT" == "all" ]; then
        echo "  - ../outputs/Heston_ot_kappa1.0/"
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
    echo "  - comprehensive_metrics.json    # All metrics (spectrum, seasonal, convergence)"
    echo "  - quality_comparison.pdf        # Generation quality comparison"
    echo "  - training_comparison.pdf       # Training metrics comparison"
    echo "  - samples_comparison.pdf        # Visual sample comparison"
    echo "  - spectrum_comparison.pdf       # Power spectrum comparison"
    echo "  - convergence_metrics.pdf       # Convergence analysis"
    echo ""
}

# Run main
main "$@"
