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
Epoch  Train           Test
       Loss    Acc     Loss    Acc
1      0.1453  95.8%   0.0923  97.46%
2      0.0521  100.0%  0.0562  98.28%
3      0.0436  97.9%   0.0432  98.73%
4      0.0301  99.0%   0.0368  98.84%
5      0.0358  99.0%   0.0359  98.81%
6      0.0083  100.0%  0.0356  98.84%
7      0.0852  97.9%   0.0342  98.82%
8      0.0379  99.0%   0.0283  99.09%
9      0.0147  100.0%  0.0258  99.18%
10     0.0388  97.9%   0.0238  99.25%
11     0.0278  99.0%   0.0232  99.16%
12     0.0060  100.0%  0.0242  99.16%
13     0.0598  97.9%   0.0224  99.33%
14     0.0142  100.0%  0.0229  99.23%
15     0.0237  99.0%   0.0207  99.41%  # Best
16     0.0281  99.0%   0.0245  99.20%
17     0.0052  100.0%  0.0220  99.31%
18     0.0677  99.0%   0.0200  99.38%
19     0.0763  96.9%   0.0218  99.34%
20     0.0115  100.0%  0.0214  99.24%
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