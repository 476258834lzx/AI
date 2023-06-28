#无需使用，测试即可
from torch import nn

def weight_init(m):
    if isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal(m.weight)
        if m.bias is not False:
            nn.init.zeros_(m.bias)
    elif isinstance(m,nn.Linear):
        nn.init.normal_(m.weight,0,0.5)
        if m.bias is not False:
            nn.init.zeros_(m.bias)
