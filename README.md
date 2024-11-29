# MNIST Digit Classification with PyTorch

![Model Tests](https://github.com/MohammedYaseen97/erav3-assgn06/actions/workflows/model_tests.yml/badge.svg)

A lightweight CNN implementation for MNIST digit classification focusing on efficient architecture design and stability during training.

## Architecture Overview

The model achieves >99.3% test accuracy with careful architectural choices and training strategies.

### Key Components

#### 1. Convolution Blocks
Custom `ConvBlock` class combining:
- 3x3 Convolution (no padding)
- Batch Normalization (eps=1e-5, momentum=0.1)
- ReLU activation
- Dropout (5%)

#### 2. Transition Block
Custom `TransitionBlock` class for dimensionality reduction:
- MaxPooling (2x2)
- 1x1 Convolution for channel reduction
- ReLU activation

### Network Structure

```
Input (28x28x1)
│
├── Initial Feature Extraction
│   ├── ConvBlock: 1 → 16 channels (26x26)
│   ├── ConvBlock: 16 → 16 channels (24x24)
│   └── ConvBlock: 16 → 32 channels (22x22)
│
├── Transition Block
│   └── 32 → 8 channels (11x11)
│
├── Feature Processing
│   ├── ConvBlock: 8 → 16 channels (9x9)
│   ├── ConvBlock: 16 → 16 channels (7x7)
│   └── Conv2d: 16 → 32 channels (5x5)
│
└── Output Block
    ├── Global Average Pooling
    └── FC: 32 → 10 classes
```

## Training Details

### Data Augmentation
```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.RandomAffine(degrees=0, shear=10)  # Shear by 10 degrees
])
```

### Training Strategy
- **Batch Size:** 128
- **Optimizer:** Adam
  - Learning Rate: 0.001
  - Betas: (0.9, 0.999)
  - Epsilon: 1e-8
- **Gradient Clipping:** max_norm=5.0
- **Learning Rate Scheduler:** ReduceLROnPlateau
  - Mode: min
  - Factor: 0.1
  - Patience: 2
- **Epochs:** 20

## Model Efficiency
- Total Parameters: 16,034
- Modern Architecture Components:
  - Batch Normalization for training stability
  - Dropout (5%) for regularization
  - Global Average Pooling to reduce parameters
  - Progressive reduction in spatial dimensions
  - Gradient clipping to prevent explosion

## Results
- Best Test Accuracy: 99.41% (Epoch 15)
- Consistent >99% accuracy after epoch 8
- Stable training without gradient explosion
- Fast convergence (reaches 97.46% in first epoch)

### Training Logs
```
epoch=1; loss=0.1453; accuracy=0.958: 100%|██████████| 469/469 [00:32<00:00, 14.38it/s]
Test set: Average loss: 0.0923, Accuracy: 9746/10000 (97.46%)

epoch=2; loss=0.0521; accuracy=1.000: 100%|██████████| 469/469 [00:32<00:00, 14.51it/s]
Test set: Average loss: 0.0562, Accuracy: 9828/10000 (98.28%)

epoch=3; loss=0.0436; accuracy=0.979: 100%|██████████| 469/469 [00:31<00:00, 14.91it/s]
Test set: Average loss: 0.0432, Accuracy: 9873/10000 (98.73%)

epoch=4; loss=0.0301; accuracy=0.990: 100%|██████████| 469/469 [00:32<00:00, 14.51it/s]
Test set: Average loss: 0.0368, Accuracy: 9884/10000 (98.84%)

epoch=5; loss=0.0358; accuracy=0.990: 100%|██████████| 469/469 [00:32<00:00, 14.45it/s]
Test set: Average loss: 0.0359, Accuracy: 9881/10000 (98.81%)

epoch=6; loss=0.0083; accuracy=1.000: 100%|██████████| 469/469 [00:31<00:00, 15.01it/s]
Test set: Average loss: 0.0356, Accuracy: 9884/10000 (98.84%)

epoch=7; loss=0.0852; accuracy=0.979: 100%|██████████| 469/469 [00:30<00:00, 15.22it/s]
Test set: Average loss: 0.0342, Accuracy: 9882/10000 (98.82%)

epoch=8; loss=0.0379; accuracy=0.990: 100%|██████████| 469/469 [00:31<00:00, 14.74it/s]
Test set: Average loss: 0.0283, Accuracy: 9909/10000 (99.09%)

epoch=9; loss=0.0147; accuracy=1.000: 100%|██████████| 469/469 [00:31<00:00, 14.69it/s]
Test set: Average loss: 0.0258, Accuracy: 9918/10000 (99.18%)

epoch=10; loss=0.0388; accuracy=0.979: 100%|██████████| 469/469 [00:31<00:00, 14.90it/s]
Test set: Average loss: 0.0238, Accuracy: 9925/10000 (99.25%)

epoch=11; loss=0.0278; accuracy=0.990: 100%|██████████| 469/469 [00:31<00:00, 15.02it/s]
Test set: Average loss: 0.0232, Accuracy: 9916/10000 (99.16%)

epoch=12; loss=0.0060; accuracy=1.000: 100%|██████████| 469/469 [00:31<00:00, 14.77it/s]
Test set: Average loss: 0.0242, Accuracy: 9916/10000 (99.16%)

epoch=13; loss=0.0598; accuracy=0.979: 100%|██████████| 469/469 [00:31<00:00, 14.83it/s]
Test set: Average loss: 0.0224, Accuracy: 9933/10000 (99.33%)

epoch=14; loss=0.0142; accuracy=1.000: 100%|██████████| 469/469 [00:32<00:00, 14.62it/s]
Test set: Average loss: 0.0229, Accuracy: 9923/10000 (99.23%)

epoch=15; loss=0.0237; accuracy=0.990: 100%|██████████| 469/469 [00:30<00:00, 15.13it/s]
Test set: Average loss: 0.0207, Accuracy: 9941/10000 (99.41%)

epoch=16; loss=0.0281; accuracy=0.990: 100%|██████████| 469/469 [00:31<00:00, 15.08it/s]
Test set: Average loss: 0.0245, Accuracy: 9920/10000 (99.20%)

epoch=17; loss=0.0052; accuracy=1.000: 100%|██████████| 469/469 [00:32<00:00, 14.41it/s]
Test set: Average loss: 0.0220, Accuracy: 9931/10000 (99.31%)

epoch=18; loss=0.0677; accuracy=0.990: 100%|██████████| 469/469 [00:31<00:00, 14.87it/s]
Test set: Average loss: 0.0200, Accuracy: 9938/10000 (99.38%)

epoch=19; loss=0.0763; accuracy=0.969: 100%|██████████| 469/469 [00:32<00:00, 14.44it/s]
Test set: Average loss: 0.0218, Accuracy: 9934/10000 (99.34%)

epoch=20; loss=0.0115; accuracy=1.000: 100%|██████████| 469/469 [00:31<00:00, 15.03it/s]
Test set: Average loss: 0.0214, Accuracy: 9924/10000 (99.24%)
```

### Key Observations
1. **Fast Initial Learning**
   - 97.46% accuracy in first epoch
   - Crosses 98% by epoch 2

2. **Stability**
   - Train accuracy fluctuates between 97-100%
   - Test accuracy steadily improves
   - No signs of overfitting

3. **Loss Trends**
   - Training loss varies between 0.005-0.15
   - Test loss consistently decreases
   - Final test loss: 0.0214

4. **Performance Peaks**
   - Best accuracy: 99.41% (Epoch 15)
   - Multiple epochs >99.3%
   - Maintains high performance till end

5. **Training Speed**
   - ~14-15 iterations/second
   - ~32 seconds per epoch
   - Total training time: ~11 minutes

## Project Structure
```
.
├── model.py                    # Model architecture
├── tests/                      # Model tests
│   ├── __init__.py
│   └── test_model.py          
├── requirements.txt            # Dependencies
└── ERAv3_Session_6.ipynb      # Training notebook
```

## Setup and Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run tests:
```bash
pytest tests/test_model.py -v
```

3. Train model: Run `ERAv3_Session_6.ipynb`

## Requirements
- Python 3.8+
- PyTorch ≥ 1.7.0
- torchvision
- numpy < 2
- tqdm
- pytest

## Model Tests
Automated tests verify:
- Parameter count < 20k
- Use of Batch Normalization
- Implementation of Dropout
- Presence of GAP/FC layer

## License
MIT

## Acknowledgments
This implementation is part of the ERA V3 course by The School of AI.