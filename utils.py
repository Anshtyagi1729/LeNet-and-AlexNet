import torch 
from torch import nn
def init_xavier(m):
    if isinstance(m,(nn.Conv2d,nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
def init_kaiming(m):
    if isinstance(m,(nn.Conv2d,nn.Linear)):
        nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
def accuracy(y_hat,y):
    return (y_hat.argmax(dim=1)==y).float().mean().item()
