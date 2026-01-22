#!/bin/bash
# Quick Start Script for MAI-UI Training Pipeline
# This script provides a one-command entry point to run the complete training flow

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print header
echo "============================================================"
echo "MAI-UI Training Pipeline - Quick Start"
echo "============================================================"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    log_error "Python3 not found. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
log_info "Python version: $PYTHON_VERSION"

# Default values
MODE="full"
CONFIG_FILE="pipeline/pipeline_config.yaml"
LLM_BASE_URL=""
SKIP_DEPS=false
RESUME=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --llm-base-url)
            LLM_BASE_URL="$2"
            shift 2
            ;;
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode <mode>           Pipeline mode: full, data, sft, eval (default: full)"
            echo "  --config <file>         Path to pipeline config (default: pipeline/pipeline_config.yaml)"
            echo "  --llm-base-url <url>    LLM API base URL (required for eval)"
            echo "  --skip-deps             Skip dependency installation"
            echo "  --resume                Resume from last failed stage"
            echo "  --help                  Show this help message"
            echo ""
            echo "Modes:"
            echo "  full    - Run complete pipeline (data + training + eval)"
            echo "  data    - Only run data preprocessing"
            echo "  sft     - Run data preprocessing + SFT training"
            echo "  eval    - Run evaluation only"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if we're in the correct directory
if [ ! -f "sft_trainer.py" ]; then
    log_error "Please run this script from the trainer directory"
    exit 1
fi

# Install dependencies
if [ "$SKIP_DEPS" = false ]; then
    log_info "Checking dependencies..."
    
    if [ ! -f "requirements.txt" ]; then
        log_warn "requirements.txt not found, skipping dependency installation"
    else
        log_info "Installing dependencies..."
        python3 -m pip install -r requirements.txt --quiet
        log_info "Dependencies installed"
    fi
else
    log_info "Skipping dependency installation"
fi

# Check for LLM_BASE_URL if needed
if [[ "$MODE" == "full" || "$MODE" == "eval" ]]; then
    if [ -z "$LLM_BASE_URL" ]; then
        if [ -z "$LLM_BASE_URL" ] && [ -z "$OPENAI_API_KEY" ]; then
            log_error "LLM_BASE_URL must be provided for evaluation"
            echo "Set it via --llm-base-url or environment variable LLM_BASE_URL"
            exit 1
        fi
    fi
    export LLM_BASE_URL="$LLM_BASE_URL"
fi

# Run based on mode or resume
if [ "$RESUME" = true ]; then
    log_info "Resuming pipeline from last failed stage..."
    python3 pipeline/orchestrator.py --config "$CONFIG_FILE" --resume
else
    case $MODE in
        full)
            log_info "Running FULL pipeline..."
            python3 pipeline/orchestrator.py --config "$CONFIG_FILE"
            ;;
        
        data)
            log_info "Running DATA preprocessing only..."
            python3 pipeline/orchestrator.py --config "$CONFIG_FILE" --stop-at data_preprocessing
            ;;
        
        sft)
            log_info "Running DATA + SFT training..."
            python3 pipeline/orchestrator.py --config "$CONFIG_FILE" --stop-at sft_training
            ;;
        
        eval)
            log_info "Running EVALUATION only..."
            python3 pipeline/orchestrator.py --config "$CONFIG_FILE" --start-from sft_evaluation
            ;;
        
        *)
            log_error "Unknown mode: $MODE"
            echo "Valid modes: full, data, sft, eval"
            exit 1
            ;;
    esac
fi

# Check exit status
if [ $? -eq 0 ]; then
    log_info "Pipeline completed successfully!"
    echo ""
    echo "============================================================"
    echo "Next steps:"
    
    case $MODE in
        data)
            echo "  1. Check processed data in dataset/processed/"
            echo "  2. Validate data: python scripts/validate_data.py -f <data_file>"
            echo "  3. Run training: ./scripts/quick_start.sh --mode sft"
            ;;
        sft)
            echo "  1. Check trained model in models/sft_model/"
            echo "  2. Run evaluation: ./scripts/quick_start.sh --mode eval"
            ;;
        eval|full)
            echo "  1. Check evaluation results in eval_logs/"
            echo "  2. View reports in batch_eval_results/"
            ;;
    esac
    
    echo "============================================================"
else
    log_error "Pipeline failed!"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check logs in pipeline_logs/"
    echo "  2. Resume from failure: python pipeline/orchestrator.py --config $CONFIG_FILE --resume"
    echo "  3. See docs/TROUBLESHOOTING.md for common issues"
    exit 1
fi
