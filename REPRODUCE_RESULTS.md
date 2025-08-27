# Reproducing Paper Results

This guide provides step-by-step instructions to reproduce all experimental results from our paper "Optimal Complexity in Lightweight Vision Transformers: A Trade-off Analysis between Representational Power and Optimization Efficiency".

## Quick Reproduction (Recommended)

If you want to quickly reproduce our key results using pre-trained models:

```bash
# 1. Download pre-trained models
bash scripts/download_pretrained.sh

# 2. Evaluate all models 
bash scripts/evaluate_all.sh /path/to/imagenet ./pretrained ./results

# 3. Generate all visualizations
bash scripts/generate_visualizations.sh /path/to/imagenet ./results ./figures
```

This will give you:
- **Table 1**: Model performance comparison
- **Figure 4**: Attention visualization comparisons  
- **Figure 5**: Complexity-performance trade-off
- **Figure 6**: t-SNE feature space analysis

## Full Training from Scratch

To completely reproduce our training pipeline:

### Prerequisites

1. **Hardware Requirements**:
   - 4 x NVIDIA GPUs (we used RTX 4090 with 24GB memory)
   - At least 64GB system RAM
   - 1TB free storage space

2. **Software Requirements**:
   ```bash
   # Install dependencies
   pip install -r requirements_academic.txt
   ```

3. **Dataset Preparation**:
   Download ImageNet-1K from [http://image-net.org/](http://image-net.org/)
   ```
   /path/to/imagenet/
   ├── train/  (1,281,167 images)
   └── val/    (50,000 images)
   ```

### Step-by-Step Training

#### Step 1: Train All Model Variants (~7 days total)

```bash
# Train all variants with our exact configuration
bash scripts/train_all_variants.sh /path/to/imagenet ./output 256 270

# Or train individual variants:
# CA/SA variants (using main.py + distributed training):
bash scripts/train_acr_ca.sh /path/to/imagenet ./output/ca 256 270
bash scripts/train_acr_sa.sh /path/to/imagenet ./output/sa 256 270

# CBAM/MultiScale variants (using acr_train.py + early stopping):
bash scripts/train_acr_cbam.sh /path/to/imagenet ./output/cbam 256 270
bash scripts/train_acr_multiscale.sh /path/to/imagenet ./output/multiscale 256 270

# Training time estimates:
# - RepViT-M0.9 (baseline): ~30 hours
# - ACR-RepViT-M0.9-CA: ~25 hours (main.py, lr=0.005)
# - ACR-RepViT-M0.9-SA: ~25 hours (main.py, lr=0.005)
# - ACR-RepViT-M0.9-CBAM: ~35 hours (acr_train.py, lr=0.001, early stopping)
# - ACR-RepViT-M0.9-MultiScale: ~20 hours (acr_train.py, lr=0.001, early stopping)
```

#### Step 2: Evaluate Trained Models

```bash
# Comprehensive evaluation
bash scripts/evaluate_all.sh /path/to/imagenet ./output ./results
```

#### Step 3: Generate Paper Figures

```bash
# Generate all visualizations
bash scripts/generate_visualizations.sh /path/to/imagenet ./results ./figures
```

## Expected Results

### Table 1: Model Performance Comparison

| Model | Top-1 (%) | Top-5 (%) | Params (M) | FLOPs (G) | Param Eff (%/M) |
|-------|-----------|-----------|------------|-----------|------------------|
| RepViT-M0.9 | 78.58 | 94.10 | 5.49 | 0.87 | 14.31 |
| ACR-RepViT-M0.9-CBAM | 73.20 | 90.60 | 7.30 | 0.92 | 10.03 |
| ACR-RepViT-M0.9-CA | 70.51 | 89.30 | 3.26 | 0.81 | 21.63 |
| ACR-RepViT-M0.9-SA | 70.71 | 89.47 | 3.26 | 0.82 | 21.69 |
| ACR-RepViT-M0.9-MultiScale | 67.43 | 87.05 | 4.07 | 0.63 | 16.57 |

**Tolerance**: Results should be within ±0.3% of reported values due to random initialization.

### Key Findings Validation

1. **Counter-intuitive Performance Drop**: CBAM variant shows 5.4pp accuracy drop
2. **Parameter Efficiency**: Original model achieves highest efficiency (14.31%/M)
3. **Optimization Difficulty**: Complex variants show training instability
4. **Feature Quality**: t-SNE visualizations show clearer class boundaries for baseline

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   ```bash
   # Reduce batch size
   bash scripts/train_all_variants.sh /path/to/imagenet ./output 128 270
   ```

2. **Slow Training**:
   ```bash
   # Use fewer epochs for quick validation
   bash scripts/train_all_variants.sh /path/to/imagenet ./output 256 100
   ```

3. **Missing Dependencies**:
   ```bash
   # Install all requirements
   pip install -r requirements_academic.txt
   
   # For CUDA issues
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Validation Checks

Verify your setup before full training:

```bash
# Quick training test (1 epoch)
python acr_train.py \
    --model acr_repvit_m0_9_cbam \
    --data-path /path/to/imagenet \
    --epochs 1 \
    --batch-size 64

# Quick evaluation test
python main.py --eval \
    --model repvit_m0_9 \
    --resume pretrained/repvit_m0_9.pth \
    --data-path /path/to/imagenet
```

## Individual Experiments

If you want to reproduce specific parts:

### Reproduce Figure 4: Attention Visualization

```bash
python generate_enhanced_visualization.py \
    --data-path /path/to/imagenet \
    --models repvit_m0_9,acr_repvit_m0_9_cbam,acr_repvit_m0_9_multiscale \
    --output-dir ./figures \
    --sample-images 5
```

### Reproduce Figure 5: Complexity-Performance Trade-off

```bash
python generate_complexity_tradeoff_final.py \
    --results-dir ./results \
    --output-path ./figures/complexity_performance_tradeoff_final.png
```

### Reproduce Figure 6: t-SNE Analysis

```bash
python tsne_visualization.py \
    --data-path /path/to/imagenet \
    --models all \
    --num-classes 10 \
    --samples-per-class 100 \
    --output-dir ./figures
```

### Reproduce Ablation Study (Table 2)

```bash
# Train attention variants
python acr_train.py --model acr_repvit_m0_9_ca --data-path /path/to/imagenet
python acr_train.py --model acr_repvit_m0_9_sa --data-path /path/to/imagenet

# Evaluate
python analyze_attention_variants.py --results-dir ./output
```

## Performance Benchmarking

### Inference Speed Measurement

```bash
# GPU throughput
python speed_gpu.py --model repvit_m0_9
python speed_gpu.py --model acr_repvit_m0_9_cbam

# Mobile deployment (requires macOS)
python export_coreml.py --model repvit_m0_9
# Measure on iPhone using Xcode
```

### Memory Usage Analysis

```bash
# Peak memory usage during training
python analyze_memory_usage.py \
    --models repvit_m0_9,acr_repvit_m0_9_cbam \
    --batch-size 256
```

## Citation

If you use our reproduction scripts or findings, please cite:

```bibtex
@article{zhang2024optimal,
  title={Optimal Complexity in Lightweight Vision Transformers: A Trade-off Analysis between Representational Power and Optimization Efficiency},
  author={Zhang, Yunan and Fan, Jingjing and Zhao, Jianguang},
  journal={The Visual Computer},
  year={2024},
  note={Under Review}
}
```

## Support

For issues with reproduction:

1. **Check our FAQ**: [REPRODUCE_FAQ.md](REPRODUCE_FAQ.md)
2. **Open an issue**: Include error logs and system information
3. **Email us**: zyn2024229134@hebiace.edu.cn

---

**Expected Total Time**: 
- Quick reproduction: ~2 hours
- Full training: ~7 days
- Individual experiments: ~1-24 hours each
