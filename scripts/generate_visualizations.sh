#!/bin/bash

# ACR-RepViT: Visualization Generation Script
# Generates all figures used in the paper

set -e

# Configuration
DATA_PATH=${1:-"/path/to/imagenet"}
RESULTS_DIR=${2:-"./results"}
OUTPUT_DIR=${3:-"./figures"}

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Generating visualizations for ACR-RepViT paper..."
echo "Data path: $DATA_PATH"
echo "Results directory: $RESULTS_DIR"
echo "Output directory: $OUTPUT_DIR"

# 1. Generate complexity-performance trade-off plot (Figure 5)
echo "=== Generating Complexity-Performance Trade-off Plot ==="
python generate_complexity_tradeoff_final.py \
    --results-dir $RESULTS_DIR \
    --output-path $OUTPUT_DIR/complexity_performance_tradeoff_final.png

# 2. Generate attention visualization comparisons (Figure 4)
echo "=== Generating Attention Visualizations ==="
python generate_enhanced_visualization.py \
    --data-path $DATA_PATH \
    --output-dir $OUTPUT_DIR \
    --num-samples 5

# 3. Generate t-SNE feature space analysis (Figure 6)
echo "=== Generating t-SNE Feature Space Analysis ==="
python tsne_visualization.py \
    --data-path $DATA_PATH \
    --output-dir $OUTPUT_DIR \
    --num-classes 10 \
    --samples-per-class 100

# 4. Generate parameter vs accuracy plot (Figure 3)
echo "=== Generating Parameter vs Accuracy Plot ==="
python params_accuracy_plot.py \
    --results-dir $RESULTS_DIR \
    --output-path $OUTPUT_DIR/params_vs_accuracy.png

# 5. Generate parameter utilization analysis (Figure 7)
echo "=== Generating Parameter Utilization Analysis ==="
python param_utilization_plot.py \
    --results-dir $RESULTS_DIR \
    --output-path $OUTPUT_DIR/param_utilization.png

# 6. Generate inference time comparison
echo "=== Generating Inference Time Comparison ==="
python speed_gpu.py \
    --models repvit_m0_9,acr_repvit_m0_9_cbam,acr_repvit_m0_9_ca,acr_repvit_m0_9_sa,acr_repvit_m0_9_multiscale \
    --output-path $OUTPUT_DIR/inference_time.png

echo "=== Visualization Generation Completed! ==="
echo "All figures saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "- complexity_performance_tradeoff_final.png"
echo "- enhanced_feature_visualization.png" 
echo "- tsne_combined.png"
echo "- params_vs_accuracy.png"
echo "- param_utilization.png"
echo "- inference_time.png"
echo ""
echo "These figures can be directly used in your paper!"

