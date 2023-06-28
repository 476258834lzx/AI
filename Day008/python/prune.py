import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
config=[
    [-1,32,1,2],
    [1,16,1,1],
    [6,24,2,2],
    [6,32,3,2],
    [6,64,4,2],
    [6,96,3,1],
    [6,160,3,2],
    [6,320,1,1]
]

class bottleneck_block(nn.Module):
    def __init__(self,c_in,i,t,c,n,s):
        super(bottleneck_block, self).__init__()
        #重复最后一次才调用步长为2的下采样
        self.n=n
        self.i=i

        _s = s if self.i==self.n-1 else 1#n为重复次数，循环索引从0开始，最后一次步长为2,通道数变为c_out
        _c=c if self.i==n-1 else c_in

        _c_in=c_in*t#升通道倍数

        self.layer=nn.Sequential(
            nn.Conv2d(c_in,_c_in,(1,1),(1,1),bias=False),
            nn.BatchNorm2d(_c_in),
            nn.ReLU6(),
            nn.Conv2d(_c_in,_c_in,(3,3),(_s,_s),padding=1,groups=_c_in,bias=False),
            nn.BatchNorm2d(_c_in),
            nn.ReLU6(),
            nn.Conv2d(_c_in,_c,(1,1),(1,1),bias=False),
            nn.BatchNorm2d(_c)
        )

    def forward(self,x):
        if self.i==self.n-1:
            return self.layer(x)
        else:
            return self.layer(x)+x

class Mobilenetv2(nn.Module):
    def __init__(self,config):
        super(Mobilenetv2, self).__init__()

        self.blocks=[]
        c_in=config[0][1]
        for t,c,n,s in config[1:]:
            for i in range(n):
                self.blocks.append(bottleneck_block(c_in,i,t,c,n,s))
            c_in=c

        self.layer=nn.Sequential(
            nn.Conv2d(3,32,(3,3),(2,2),(1,1),bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
            *self.blocks,
            nn.Conv2d(320,1280,(1,1),(1,1),bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(),
            nn.AvgPool2d((7,7),1),
            nn.Conv2d(1280,10,(1,1),(1,1),bias=False)
        )
    def forward(self,x):
        return self.layer(x)

if __name__ == '__main__':
    net=Mobilenetv2(config)

    #按层剪枝，不常用
    # module=net.layer[0]
    # print(list(module.named_parameters()))#查看网络权重缓存区，如果剪枝精度下降过大了可恢复
    # print(list(module.named_buffers()))#存放剪枝需要的mask的缓存区
    #随机非结构剪枝           需要剪的参数名 weight or bias   剪枝幅度 小数剪%多少，整数剪几个
    # prune.random_unstructured(module,name="weight",amount=0.3)#权重矩阵和掩码矩阵挨个元素相乘,可叠加，在上次的基础上再次剪枝
    # print(list(module.named_buffers()))
    # print(module.weight)#剪枝之后的权重
    # print(list(module.named_parameters()))
    # print(module._forward_pre_hooks)
    # prune.ln_structured(module,name="weight",amount=0.5,n=2,dim=0)
    # for hook in module._forward_pre_hooks.values():#查看剪枝记录
    #     if hook._tensor_name=="weight":
    #         break
    # print(hook)
    # print(net.state_dict().keys())#查看新增mask缓存区
    # prune.remove(module,"weight")#清空上一次named_parameters缓存，把当前剪枝后的权重存进去



    #整个模型局部剪枝
    # for name,module in net.named_modules():
    #     if isinstance(module,torch.nn.Conv2d):
    #         prune.l1_unstructured(module,name="weight",amount=0.2)
    #     elif isinstance(module,torch.nn.Linear):
    #         prune.l1_unstructured(module,name="weight",amount=0.4)
    #
    # print(dict(net.named_buffers()).keys())

    #整个模型全局剪枝，不常用
    # parameters_to_prune=(
    #     (net.layer)
    # )
    # prune.global_unstructured(
    #     parameters_to_prune,
    #     pruning_method=prune.l1_unstructured,
    #     amount=0.2
    # )