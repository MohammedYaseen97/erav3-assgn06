# MNIST Digit Classification with PyTorch

![Model Tests](https://github.com/MohammedYaseen97/erav3-assgn06/actions/workflows/model_tests.yml/badge.svg)

A lightweight CNN implementation for MNIST digit classification focusing on efficient architecture design and data augmentation strategies.

## Architecture Overview

The model achieves >99% test accuracy while maintaining a small parameter footprint through careful architectural choices.

### Key Components

#### 1. Convolution Blocks
Custom `ConvBlock` class combining:
- 3x3 Convolution
- Batch Normalization
- ReLU activation
- Dropout (5%)

#### 2. Transition Blocks
Custom `TransitionBlock` class for dimensionality reduction:
- MaxPooling (2x2)
- 1x1 Convolution for channel reduction
- ReLU activation

### Network Structure

1. **Initial Feature Extraction**
   - Three ConvBlocks (1→8→16→32 channels)
   - Maintains spatial dimensions (28x28)

2. **First Transition**
   - Reduces spatial dimensions: 28x28 → 14x14
   - Reduces channels: 32 → 4

3. **Mid-Level Features**
   - Three ConvBlocks (4→8→16→32 channels)
   - Maintains 14x14 spatial dimensions

4. **Second Transition**
   - Further reduces dimensions: 14x14 → 7x7
   - Reduces channels: 32 → 4

5. **Output Block**
   - Final convolution: 4 → 8 channels
   - Global Average Pooling
   - Fully Connected layer: 8 → 10 classes

## Training Strategy

### Data Augmentation

```
Input (28x28x1)
│
├── Initial Feature Extraction
│ ├── ConvBlock: 1 → 8 channels
│ ├── ConvBlock: 8 → 16 channels
│ └── ConvBlock: 16 → 32 channels
│
├── First Transition (28x28 → 14x14)
│ └── 32 → 4 channels
│
├── Mid-Level Features
│ ├── ConvBlock: 4 → 8 channels
│ ├── ConvBlock: 8 → 16 channels
│ └── ConvBlock: 16 → 32 channels
│
├── Second Transition (14x14 → 7x7)
│ └── 32 → 4 channels
│
└── Output Block
├── Conv: 4 → 8 channels
├── Global Average Pooling
└── FC: 8 → 10 classes
```

## Training Details

### Data Augmentation

```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
])
```

### Hyperparameters
- **Batch Size:** 64
- **Optimizer:** SGD
  - Learning Rate: 0.01
  - Momentum: 0.9
- **Epochs:** 20

## Model Efficiency
- Total Parameters: <20,000
- Modern Architecture Components:
  - Batch Normalization for training stability
  - Dropout (5%) for regularization
  - Global Average Pooling to reduce parameters
  - 1x1 convolutions for efficient channel reduction

## Project Structure

```
.
├── model.py # Model architecture
├── tests/ # Model tests
│ ├── init.py
│ └── test_model.py
├── requirements.txt # Dependencies
└── ERAv3_Session_6.ipynb # Training notebook
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