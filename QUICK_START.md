# Quick Start Guide

This guide helps you get ACR-RepViT running in 5 minutes.

## 🚀 Option 1: Quick Evaluation (Recommended)

If you just want to see our results:

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ACR-RepViT.git
cd ACR-RepViT

# 2. Install dependencies  
pip install -r requirements_academic.txt

# 3. Download pre-trained models
bash scripts/download_pretrained.sh

# 4. Quick evaluation (uses CIFAR-10 for demo)
python main.py --eval --model repvit_m0_9 --dataset cifar10 --data-path ./data

# 5. Generate a sample visualization
python generate_enhanced_visualization.py --demo
```

## 🔬 Option 2: Full Reproduction

For complete paper reproduction:

```bash
# 1. Prepare ImageNet-1K dataset
# Download from http://image-net.org/ to /path/to/imagenet/

# 2. Train all models (takes ~1 week)
bash scripts/train_all_variants.sh /path/to/imagenet

# 3. Evaluate and generate figures
bash scripts/evaluate_all.sh /path/to/imagenet ./output ./results
bash scripts/generate_visualizations.sh /path/to/imagenet ./results ./figures
```

## 📊 Expected Results

| Model | Top-1 Acc (%) | Key Finding |
|-------|---------------|-------------|
| RepViT-M0.9 (Baseline) | 78.58 | Optimal performance |
| + CBAM | 73.20 | **5.4pp drop!** |
| + MultiScale | 67.43 | **11.2pp drop!** |

**Key Insight**: More complexity ≠ Better performance in lightweight models!

## 🛠 Troubleshooting

**Error: Model not found**
```bash
# Make sure models are registered
python -c "from models import repvit_m0_9; print('✓ Models loaded successfully')"
```

**Error: Out of memory**
```bash
# Reduce batch size
export CUDA_VISIBLE_DEVICES=0
python main.py --batch-size 64  # instead of 256
```

**Error: Dataset not found**
```bash
# Use CIFAR-10 for quick testing
python main.py --dataset cifar10 --data-path ./data --download
```

## 📚 Core Files

| File | Purpose |
|------|---------|
| `main.py` | Original RepViT training/eval |
| `acr_train.py` | ACR variant training/eval |
| `models/acr_repvit_*.py` | Our model implementations |
| `scripts/train_all_variants.sh` | Complete training pipeline |
| `generate_*.py` | Visualization scripts |

## 🎯 Paper Experiments

### Reproduce Figure 4 (Attention Maps)
```bash
python generate_enhanced_visualization.py --data-path /path/to/imagenet
```

### Reproduce Figure 5 (Complexity Trade-off)  
```bash
python generate_complexity_tradeoff_final.py --results-dir ./results
```

### Reproduce Table 1 (Main Results)
```bash
bash scripts/evaluate_all.sh /path/to/imagenet ./output ./results
```

## 💡 Quick Tips

1. **Start with CIFAR-10**: Much faster than ImageNet for initial testing
2. **Use pre-trained models**: Skip training if you just want to see results  
3. **Check GPU memory**: Our models need 8-24GB GPU memory for training
4. **Read logs**: All training logs are saved to `./output/*/log.txt`

## 🔗 Important Links

- **Paper**: [arXiv:2024.XXXX.XXXXX](https://arxiv.org/abs/2024.XXXX.XXXXX)
- **Models**: [GitHub Releases](https://github.com/YOUR_USERNAME/ACR-RepViT/releases)
- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/ACR-RepViT/issues)
- **Original RepViT**: [THU-MIG/RepViT](https://github.com/THU-MIG/RepViT)

## ✅ Verification

Test your setup:

```bash
# Quick system check
python -c "
import torch
import timm
from models import repvit_m0_9
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ TIMM: {timm.__version__}')
print(f'✓ CUDA: {torch.cuda.is_available()}')
print('✓ All dependencies working!')
"
```

---

**🎉 Ready to explore? Start with Option 1 for a quick demo!**

