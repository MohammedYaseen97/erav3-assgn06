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
- Best Test Accuracy: 99.41%
- Consistent >99% accuracy after epoch 8
- Stable training without gradient explosion
- Fast convergence (reaches 97% in first epoch)

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