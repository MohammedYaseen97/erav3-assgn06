import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Net

def test_parameter_count():
    model = Net()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, f'Model has {total_params} parameters, should be < 20000'

def test_batch_normalization():
    model = Net()
    has_bn = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert has_bn, 'Model must use Batch Normalization'

def test_dropout():
    model = Net()
    has_dropout = any(isinstance(m, nn.Dropout2d) for m in model.modules())
    assert has_dropout, 'Model must use Dropout'

def test_gap_or_fc():
    model = Net()
    has_gap = any(isinstance(m, nn.AdaptiveAvgPool2d) for m in model.modules())
    has_fc = any(isinstance(m, nn.Linear) for m in model.modules())
    assert has_gap or has_fc, 'Model must use either GAP or FC layer'

if __name__ == '__main__':
    test_parameter_count()
    test_batch_normalization()
    test_dropout()
    test_gap_or_fc()
    print('\nAll tests passed successfully!') 