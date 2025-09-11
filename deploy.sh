#!/bin/bash

# Enterprise Arbor Model Deployment Automation
# Complete deployment script for 200B-400B parameter models

set -e  # Exit on any error

# Configuration
MODEL_SIZE="${1:-200b}"
COMMAND="${2:-create}"
WORLD_SIZE="${3:-8}"
OUTPUT_DIR="${4:-./models/arbor-${MODEL_SIZE}}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Enterprise Arbor Deployment Script${NC}"
echo -e "${BLUE}Model Size: ${MODEL_SIZE}, Command: ${COMMAND}${NC}"
echo "=================================="

# Validate inputs
if [[ "$MODEL_SIZE" != "200b" && "$MODEL_SIZE" != "400b" ]]; then
    echo -e "${RED}❌ Error: Model size must be '200b' or '400b'${NC}"
    exit 1
fi

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}⚠️  Warning: nvidia-smi not found. CUDA may not be available.${NC}"
else
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo -e "${GREEN}🔍 Found ${GPU_COUNT} GPU(s)${NC}"
    
    if [ "$GPU_COUNT" -lt 8 ] && [ "$MODEL_SIZE" = "200b" ]; then
        echo -e "${YELLOW}⚠️  Warning: 200B model recommended for 8+ GPUs, found ${GPU_COUNT}${NC}"
    fi
    
    if [ "$GPU_COUNT" -lt 16 ] && [ "$MODEL_SIZE" = "400b" ]; then
        echo -e "${YELLOW}⚠️  Warning: 400B model recommended for 16+ GPUs, found ${GPU_COUNT}${NC}"
    fi
fi

# Check Python dependencies
echo -e "${BLUE}📦 Checking dependencies...${NC}"
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || {
    echo -e "${RED}❌ PyTorch not found. Please install: pip install torch${NC}"
    exit 1
}

python -c "import transformers; print(f'Transformers: {transformers.__version__}')" || {
    echo -e "${YELLOW}⚠️  Transformers not found. Installing...${NC}"
    pip install transformers
}

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints
mkdir -p models

# Execute command
case "$COMMAND" in
    "create")
        echo -e "${GREEN}🏗️  Creating ${MODEL_SIZE} model...${NC}"
        python scripts/enterprise_deploy.py create \
            --model-size "$MODEL_SIZE" \
            --output-dir "$OUTPUT_DIR" \
            --log-level INFO 2>&1 | tee "logs/create_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S).log"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Model created successfully at ${OUTPUT_DIR}${NC}"
        else
            echo -e "${RED}❌ Model creation failed${NC}"
            exit 1
        fi
        ;;
        
    "train")
        echo -e "${GREEN}🎯 Training ${MODEL_SIZE} model...${NC}"
        
        # Set training parameters based on model size
        if [ "$MODEL_SIZE" = "200b" ]; then
            LEARNING_RATE="3e-4"
            MAX_STEPS="100000"
            CHECKPOINT_DIR="./checkpoints/arbor-200b"
        else
            LEARNING_RATE="1e-4"
            MAX_STEPS="200000"
            CHECKPOINT_DIR="./checkpoints/arbor-400b"
        fi
        
        mkdir -p "$CHECKPOINT_DIR"
        
        python scripts/enterprise_deploy.py train \
            --model-size "$MODEL_SIZE" \
            --world-size "$WORLD_SIZE" \
            --learning-rate "$LEARNING_RATE" \
            --batch-size 1 \
            --max-steps "$MAX_STEPS" \
            --checkpoint-dir "$CHECKPOINT_DIR" \
            --log-level INFO 2>&1 | tee "logs/train_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S).log"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Training completed successfully${NC}"
        else
            echo -e "${RED}❌ Training failed${NC}"
            exit 1
        fi
        ;;
        
    "serve")
        echo -e "${GREEN}🌐 Deploying inference server for ${MODEL_SIZE} model...${NC}"
        
        # Check if model exists
        MODEL_PATH="./checkpoints/arbor-${MODEL_SIZE}"
        if [ ! -d "$MODEL_PATH" ]; then
            MODEL_PATH="./models/arbor-${MODEL_SIZE}"
            if [ ! -d "$MODEL_PATH" ]; then
                echo -e "${RED}❌ Model not found at ${MODEL_PATH}${NC}"
                echo -e "${YELLOW}💡 Run: $0 ${MODEL_SIZE} create${NC}"
                exit 1
            fi
        fi
        
        # Set inference parameters
        if [ "$MODEL_SIZE" = "200b" ]; then
            BATCH_SIZE="4"
        else
            BATCH_SIZE="2"
        fi
        
        python scripts/enterprise_deploy.py serve \
            --model-size "$MODEL_SIZE" \
            --model-path "$MODEL_PATH" \
            --batch-size "$BATCH_SIZE" \
            --test-inference \
            --log-level INFO 2>&1 | tee "logs/serve_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S).log"
        ;;
        
    "benchmark")
        echo -e "${GREEN}📈 Benchmarking ${MODEL_SIZE} model...${NC}"
        
        # Check if model exists
        MODEL_PATH="./checkpoints/arbor-${MODEL_SIZE}"
        if [ ! -d "$MODEL_PATH" ]; then
            MODEL_PATH="./models/arbor-${MODEL_SIZE}"
            if [ ! -d "$MODEL_PATH" ]; then
                echo -e "${RED}❌ Model not found at ${MODEL_PATH}${NC}"
                echo -e "${YELLOW}💡 Run: $0 ${MODEL_SIZE} create${NC}"
                exit 1
            fi
        fi
        
        # Set benchmark parameters
        if [ "$MODEL_SIZE" = "200b" ]; then
            NUM_SAMPLES="100"
        else
            NUM_SAMPLES="50"
        fi
        
        python scripts/enterprise_deploy.py benchmark \
            --model-size "$MODEL_SIZE" \
            --model-path "$MODEL_PATH" \
            --num-samples "$NUM_SAMPLES" \
            --log-level INFO 2>&1 | tee "logs/benchmark_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S).log"
        ;;
        
    "demo")
        echo -e "${GREEN}🧪 Running demo for ${MODEL_SIZE} model...${NC}"
        
        # Quick demo pipeline
        echo -e "${BLUE}Step 1: Creating model...${NC}"
        python scripts/enterprise_deploy.py create \
            --model-size "$MODEL_SIZE" \
            --output-dir "./demo/arbor-${MODEL_SIZE}" \
            --log-level INFO
        
        echo -e "${BLUE}Step 2: Testing inference...${NC}"
        python scripts/enterprise_deploy.py serve \
            --model-size "$MODEL_SIZE" \
            --model-path "./demo/arbor-${MODEL_SIZE}" \
            --test-inference \
            --demo \
            --log-level INFO
        
        echo -e "${GREEN}✅ Demo completed successfully${NC}"
        ;;
        
    "clean")
        echo -e "${YELLOW}🧹 Cleaning up...${NC}"
        read -p "This will delete all models, checkpoints, and logs. Continue? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf models/ checkpoints/ logs/ demo/
            echo -e "${GREEN}✅ Cleanup completed${NC}"
        else
            echo -e "${BLUE}ℹ️  Cleanup cancelled${NC}"
        fi
        ;;
        
    *)
        echo -e "${RED}❌ Unknown command: ${COMMAND}${NC}"
        echo ""
        echo "Usage: $0 <model_size> <command> [world_size] [output_dir]"
        echo ""
        echo "Model Sizes:"
        echo "  200b    - 200 billion parameter model"
        echo "  400b    - 400 billion parameter model"
        echo ""
        echo "Commands:"
        echo "  create     - Create and save model"
        echo "  train      - Train model with distributed setup"
        echo "  serve      - Deploy inference server"
        echo "  benchmark  - Benchmark model performance"
        echo "  demo       - Run complete demo pipeline"
        echo "  clean      - Clean up all files"
        echo ""
        echo "Examples:"
        echo "  $0 200b create"
        echo "  $0 400b train 16"
        echo "  $0 200b serve"
        echo "  $0 400b benchmark"
        echo "  $0 200b demo"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}🎉 Operation completed successfully!${NC}"
echo -e "${BLUE}📊 Check logs in ./logs/ directory${NC}"

# Display system information
echo ""
echo -e "${BLUE}📋 System Information:${NC}"
echo "Date: $(date)"
echo "Python: $(python --version 2>&1 | head -1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not found')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'Not available')"
if command -v nvidia-smi &> /dev/null; then
    echo "GPUs: $(nvidia-smi --list-gpus | wc -l)"
    echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)MB (per GPU)"
fi
