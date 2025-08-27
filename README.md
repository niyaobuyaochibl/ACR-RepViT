# Optimal Complexity in Lightweight Vision Transformers: A Trade-off Analysis between Representational Power and Optimization Efficiency

[![arXiv](https://img.shields.io/badge/arXiv-2024.XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2024.XXXX.XXXXX)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Note**: This code repository is directly related to the manuscript currently submitted to *The Visual Computer*. If you find our work helpful for your research, please consider citing our paper.

## Abstract

The deployment of deep learning models on resource-constrained edge devices necessitates a critical balance between performance and complexity. This study systematically challenges the prevailing assumption that enhancing lightweight vision transformers with sophisticated modules invariably improves performance. By investigating the impact of structural enhancements on the state-of-the-art lightweight Vision Transformer, RepViT-M0.9, our experiments on ImageNet-1K reveal that increasing structural complexity can significantly degrade accuracy and parameter efficiency. Visualizations and feature space analysis suggest that excessive complexity within a lightweight model impairs feature representations and introduces optimization challenges. We propose the Representation-Optimization Trade-off Theory, which models performance as a balance between representational power and optimization cost. Our findings demonstrate that an optimal complexity level exists for lightweight models, beyond which performance deteriorates. This work highlights the importance of structural simplicity and parameter efficiency in developing effective AI solutions for edge devices.

**Source code and pre-trained models are publicly available at this repository and archived under DOI [Zenodo DOI to be added].**

## Key Contributions

- **🔍 Systematic Analysis**: First comprehensive study challenging the "more complexity = better performance" paradigm in lightweight Vision Transformers
- **📊 Counter-intuitive Findings**: Empirical evidence that structural enhancements (CBAM, Multi-scale) can degrade performance in lightweight models
- **🧮 Theoretical Framework**: Novel Representation-Optimization Trade-off Theory explaining optimal complexity levels
- **💡 Design Principles**: Actionable guidelines for developing efficient AI solutions for edge devices
- **🎯 Applications**: Direct relevance to computer graphics, AR/VR, and real-time rendering applications

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/niyaobuyaochibl/ACR-RepViT.git
cd ACR-RepViT

# Create conda environment
conda create -n acr_repvit python=3.8
conda activate acr_repvit

# Install dependencies
pip install -r requirements.txt
```

### Pre-trained Models

Download our pre-trained models and experimental checkpoints:

| Model | Top-1 Acc | Params | FLOPs | Download |
|-------|-----------|---------|--------|----------|
| RepViT-M0.9 (Baseline) | 78.58% | 5.49M | 0.87G | [model](https://github.com/niyaobuyaochibl/ACR-RepViT/releases/download/v1.0/repvit_m0_9_baseline.pth) |
| ACR-RepViT-M0.9-CBAM | 73.20% | 7.30M | 0.92G | [model](https://github.com/niyaobuyaochibl/ACR-RepViT/releases/download/v1.0/acr_repvit_m0_9_cbam.pth) |
| ACR-RepViT-M0.9-CA | 70.51% | 3.26M | 0.81G | [model](https://github.com/niyaobuyaochibl/ACR-RepViT/releases/download/v1.0/acr_repvit_m0_9_ca.pth) |
| ACR-RepViT-M0.9-SA | 70.71% | 3.26M | 0.82G | [model](https://github.com/niyaobuyaochibl/ACR-RepViT/releases/download/v1.0/acr_repvit_m0_9_sa.pth) |
| ACR-RepViT-M0.9-MultiScale | 67.43% | 4.07M | 0.63G | [model](https://github.com/niyaobuyaochibl/ACR-RepViT/releases/download/v1.0/acr_repvit_m0_9_multiscale.pth) |

### Quick Evaluation

```bash
# Evaluate RepViT-M0.9 baseline
python main.py --eval --model repvit_m0_9 \
    --resume pretrained/repvit_m0_9_baseline.pth \
    --data-path /path/to/imagenet

# Evaluate ACR-RepViT-M0.9-CBAM variant
python acr_train.py --eval --model acr_repvit_m0_9_cbam \
    --resume checkpoints/acr_repvit_m0_9_cbam.pth \
    --data-path /path/to/imagenet
```

## Dataset Preparation

### ImageNet-1K

Download ImageNet-1K from [http://image-net.org/](http://image-net.org/). The directory structure should be:

```
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

## Training from Scratch

### Train All Variants

Run our comprehensive training pipeline to reproduce all experimental results:

```bash
# Train all ACR-RepViT variants
bash scripts/train_all_variants.sh

# Train specific variants
# For CA/SA (using main.py):
bash scripts/train_acr_ca.sh /path/to/imagenet

# For CBAM/MultiScale (using acr_train.py):
bash scripts/train_acr_cbam.sh /path/to/imagenet
```

### Training Configuration

Models use two different training pipelines for optimal performance:

**CA/SA variants (using main.py):**
- **Optimizer**: AdamW with learning rate 0.005
- **Scheduler**: Cosine decay with 5-epoch warmup (warmup_lr=1e-5)
- **Training Duration**: 270 epochs
- **Distributed Training**: torch.distributed.launch

**CBAM/MultiScale variants (using acr_train.py):**
- **Optimizer**: AdamW with learning rate 0.001
- **Warmup**: 20 epochs
- **Training Duration**: 270 epochs (with early stopping, patience=15)
- **Additional Features**: Early stopping for optimal convergence

**Common Settings:**
- **Batch Size**: 256
- **Weight Decay**: 0.05
- **Data Augmentation**: Standard ImageNet augmentation

## Reproducing Paper Results

### Main Experimental Results (Table 1)

```bash
# Reproduce Table 1: Model Performance Comparison
python evaluate_all_models.py --data-path /path/to/imagenet --output results/table1.csv
```

### Attention Visualization (Figure 4)

```bash
# Generate attention visualization comparisons
python generate_enhanced_visualization.py \
    --model-configs configs/all_variants.yaml \
    --data-path /path/to/imagenet \
    --output-dir visualization_output/
```

### Feature Space Analysis (Figure 6)

```bash
# Generate t-SNE visualizations
python tsne_visualization.py \
    --models repvit_m0_9,acr_repvit_m0_9_cbam,acr_repvit_m0_9_multiscale \
    --data-path /path/to/imagenet \
    --output-dir analysis_output/
```

### Complexity-Performance Trade-off (Figure 5)

```bash
# Generate complexity-performance analysis
python generate_complexity_tradeoff_final.py \
    --results-dir output/ \
    --output-path figures/complexity_tradeoff_final.png
```

## Model Architectures

### ACR-RepViT Variants

Our study investigates the following structural enhancements:

1. **ACR-RepViT-M0.9-CBAM**: Integration of Convolutional Block Attention Module
2. **ACR-RepViT-M0.9-CA**: Channel Attention only
3. **ACR-RepViT-M0.9-SA**: Spatial Attention only  
4. **ACR-RepViT-M0.9-MultiScale**: Multi-scale convolution with parallel 3×3 and 5×5 paths

### Code Structure

```
models/
├── repvit.py                  # Baseline RepViT implementation
├── acr_repvit_cbam.py        # CBAM variant
├── acr_repvit_ca.py          # Channel attention variant
├── acr_repvit_sa.py          # Spatial attention variant
└── acr_repvit_multiscale.py  # Multi-scale variant
```

## Theoretical Framework

### Representation-Optimization Trade-off Theory

Our core theoretical contribution models performance as:

```
P(M, D) = R(M) - λ(M) · O(M, D)
```

Where:
- `R(M)`: Representational capability 
- `O(M, D)`: Optimization difficulty
- `λ(M)`: Optimization difficulty coefficient

For implementation details, see `theory/trade_off_analysis.py`.

## Applications in Computer Graphics & VR

Our findings have direct implications for:

- **Real-time AR/VR**: Efficient object recognition and tracking for anchoring virtual objects
- **Graphics Rendering**: Deep learning-based denoising and super-resolution with millisecond requirements  
- **Mobile Graphics**: Optimized models for next-generation immersive experiences
- **Edge Computing**: Deployment on resource-constrained graphics hardware

## Experimental Scripts

### Comprehensive Evaluation

```bash
# Run complete experimental pipeline
bash scripts/run_full_experiments.sh

# Individual experiment components
bash scripts/train_baseline.sh          # Train RepViT-M0.9 baseline
bash scripts/train_cbam_variant.sh      # Train CBAM variant
bash scripts/train_multiscale_variant.sh # Train MultiScale variant
bash scripts/evaluate_all.sh            # Evaluate all models
bash scripts/generate_visualizations.sh  # Generate all figures
```

### Inference Speed Benchmarking

```bash
# Measure inference speed on GPU
python speed_gpu.py --model repvit_m0_9
python speed_gpu.py --model acr_repvit_m0_9_cbam

# Mobile deployment (Core ML)
python export_coreml.py --model repvit_m0_9 --ckpt pretrained/repvit_m0_9.pth
```

## Results Summary

| Model | Top-1 Acc | Params | FLOPs | Param Eff | Key Finding |
|-------|-----------|---------|--------|-----------|-------------|
| RepViT-M0.9 | **78.58%** | 5.49M | 0.87G | **14.31%/M** | Optimal baseline |
| +CBAM | 73.20% ↓5.4pp | 7.30M | 0.92G | 10.03%/M | Complexity hurts |
| +MultiScale | 67.43% ↓11.2pp | 4.07M | 0.63G | 16.57%/M | Lower params, worse accuracy |
| +CA only | 70.51% ↓8.1pp | 3.26M | 0.81G | **21.63%/M** | High efficiency, low accuracy |
| +SA only | 70.71% ↓7.9pp | 3.26M | 0.82G | **21.69%/M** | High efficiency, low accuracy |

**Key Insight**: Structural simplicity achieves optimal performance-efficiency trade-off in lightweight models.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you find our work helpful in your research, please consider citing:

```bibtex
@article{zhang2024optimal,
  title={Optimal Complexity in Lightweight Vision Transformers: A Trade-off Analysis between Representational Power and Optimization Efficiency},
  author={Zhang, Yunan and Fan, Jingjing and Zhao, Jianguang},
  journal={The Visual Computer},
  year={2024},
  note={Under Review}
}
```

## Acknowledgments

- **Original RepViT**: Built upon the excellent work by [Wang et al.](https://github.com/THU-MIG/RepViT)
- **ImageNet Dataset**: We thank the ImageNet team for providing the benchmark dataset
- **Computing Resources**: Experiments conducted at Hebei University of Architecture
- **Funding**: Supported by Zhangjiakou Municipal Science and Technology Program (Grant 2311010A) and Hebei Collaborative Innovation Center of Green Buildings

## Contact

- **Yunan Zhang**: zyn2024229134@hebiace.edu.cn
- **Jingjing Fan**: fjj1960@hebiace.edu.cn (Corresponding Author)
- **Jianguang Zhao**: zjg1344@hebiace.edu.cn

---

**⭐ If you find this work useful, please star our repository and cite our paper!**
