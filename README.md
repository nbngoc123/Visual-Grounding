# Visual Grounding with TransVG

End-to-End Visual Grounding using Transformer architecture.

## Project Structure

```
visual_grounding_transvg/
├── base/                       # Base classes
│   ├── base_model.py           # BaseModel
│   ├── base_data_loader.py     # BaseDataLoader
│   └── base_trainer.py         # BaseTrainer
├── data_loader/
│   ├── datasets.py             # RefCOCODataset
│   └── data_loaders.py         # RefCOCODataLoader
├── model/
│   ├── model.py                # TransVG, MLP, VLTransformer
│   ├── loss.py                 # VisualGroundingLoss
│   └── metric.py               # compute_iou
├── trainer/
│   └── trainer.py              # Trainer
├── logger/                     # Logging utilities
├── utils/                      # Visualization
├── notebooks/                  # Experiments
├── data/                       # Data directory
├── checkpoints/                # Model weights
├── outputs/                    # Logs & predictions
├── config.py                   # Configuration
├── train.py                    # Training entry
└── test.py                     # Inference entry
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Dowload dataset. save ./data
!gdown 1m_vl7oHmxBM6SXeYSoZ5hQWH2XncIOZ1
!unzip refcoco.zip -d data

# Train ViT backbone
python train.py vit

# Train ConvNeXt backbone
python train.py convnext

# Train with custom hyperparameters
python train.py vit --epochs 30 --batch_size 64 --lr 0.0005

# Resume training from checkpoint
python train.py vit --resume checkpoints/vit/model_best.pth

# Test
python test.py vit -c checkpoints/vit/model_best.pth

# Test with visualization
python test.py convnext -c checkpoints/convnext/model_best.pth -v
```

## Model

- **Vision Backbone**: ViT-Base / ConvNeXtV2-Base
- **Text Encoder**: BERT-base
- **Fusion**: VL Transformer with learnable [REG] token
- **Loss**: L1 + 2.0 × GIoU
