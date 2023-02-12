import torch
from torch.nn import functional as F
import numpy as np

def compare_cosion(vector1,vector2):
    vector1_normal=F.normalize(vector1)
    vector2_normal=F.normalize(vector2)
    cos=torch.matmul(vector1_normal,vector2_normal.t())
    return cos