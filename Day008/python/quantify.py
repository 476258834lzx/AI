import torch
#数据量化
# input=torch.randn(2,2)
# print(input)
# Q=torch.quantize_per_tensor(input,0.025,zero_point=0,dtype=torch.qint8)
# print(Q)
# print(Q.dtype)
# P=Q.dequantize()
# print(P)
# print(P.dtype)

#Eager Mode模式动态模型量化
# -*- coding:utf-8 -*-
# Author:凌逆战 | Never
# Date: 2022/10/17
# """
# 只量化权重，不量化激活
# """
# import torch
# from torch import nn
#
# class DemoModel(torch.nn.Module):
#     def __init__(self):
#         super(DemoModel, self).__init__()
#         self.conv = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(1,1))
#         self.relu = nn.ReLU()
#         self.fc = torch.nn.Linear(2, 2)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         x = self.fc(x)
#         return x
#
#
# if __name__ == "__main__":
#     model_fp32 = DemoModel()
#     # 创建一个量化的模型实例
#     model_int8 = torch.quantization.quantize_dynamic(
#         model=model_fp32,  # 原始模型
#         qconfig_spec={torch.nn.Linear},  # 要动态量化的NN算子
#         dtype=torch.qint8)  # 将权重量化为：float16 \ qint8
#
#     print(model_fp32)
#     print(model_int8)
#
#     # 运行模型
#     input_fp32 = torch.randn(1,1,2, 2)
#     output_fp32 = model_fp32(input_fp32)
#     print(output_fp32)
#
#     output_int8 = model_int8(input_fp32)
#     print(output_int8)

#FX模式动态模型量化
# import torch
# from torch import nn
#
# # toy model
# m = nn.Sequential(
#   nn.Conv2d(2, 64, (8,)),
#   nn.ReLU(),
#   nn.Linear(16,10),
#   nn.LSTM(10, 10))
#
# m.eval()
# input_fp32 = torch.randn(1,1,2, 2)
# from torch.quantization import quantize_fx
# qconfig_dict = {"": torch.quantization.default_dynamic_qconfig}  # 空键表示应用于所有模块的默认值
# model_prepared = quantize_fx.prepare_fx(m, qconfig_dict,example_inputs=(input_fp32,))
# model_quantized = quantize_fx.convert_fx(model_prepared)

#静态量化
# import torch
# from torch import nn
#
# class F32Model(nn.Module):
#     def __init__(self):
#         super(F32Model, self).__init__()
#         self.fc = nn.Linear(3, 2,bias=False)
#         self.relu = nn.ReLU(inplace=False)
#
#     def forward(self, x):
#         x = self.fc(x)
#         x = self.relu(x)
#         return x
#
# model_fp32 = F32Model()
# print(model_fp32)
# # F32Model(
# #   (fc): Linear(in_features=3, out_features=2, bias=False)
# #   (relu): ReLU()
# # )
# model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['fc', 'relu']])
# print(model_fp32_fused)
# F32Model(
#   (fc): LinearReLU(
#     (0): Linear(in_features=3, out_features=2, bias=False)
#     (1): ReLU()
#   )
#   (relu): Identity()
# )
#

#完整的静态量化
# -*- coding:utf-8 -*-
# Author:凌逆战 | Never
# Date: 2022/10/17
"""
权重和激活都会被量化
"""

import torch
from torch import nn


# 定义一个浮点模型，其中一些层可以被静态量化
class F32Model(torch.nn.Module):
    def __init__(self):
        super(F32Model, self).__init__()
        self.quant = torch.quantization.QuantStub()  # QuantStub: 转换张量从浮点到量化
        self.conv = nn.Conv2d(1, 1, 1)
        self.fc = nn.Linear(2, 2, bias=False)
        self.relu = nn.ReLU()
        self.dequant = torch.quantization.DeQuantStub()  # DeQuantStub: 将量化张量转换为浮点

    def forward(self, x):
        x = self.quant(x)  # 手动指定张量: 从浮点转换为量化
        x = self.conv(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dequant(x)  # 手动指定张量: 从量化转换到浮点
        return x


model_fp32 = F32Model()
model_fp32.eval()  # 模型必须设置为eval模式，静态量化逻辑才能工作

# 1、如果要部署在ARM上；果要部署在x86 server上 ‘fbgemm’
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# 2、在适用的情况下，将一些层进行融合，可以加速
# 常见的融合包括在：DEFAULT_OP_LIST_TO_FUSER_METHOD
model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['fc', 'relu']])

# 3、准备模型，插入observers，观察 activation 和 weight
model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

# 4、代表性数据集，获取数据的分布特点，来更好的计算activation的 scale 和 zp
input_fp32 = torch.randn(1, 1, 2, 2)  # (batch_size, channel, W, H)
model_fp32_prepared(input_fp32)

# 5、量化模型
model_int8 = torch.quantization.convert(model_fp32_prepared)

# 运行模型，相关计算将在int8中进行
output_fp32 = model_fp32(input_fp32)
output_int8 = model_int8(input_fp32)
print(output_fp32)
# tensor([[[[0.6315, 0.0000],
#           [0.2466, 0.0000]]]], grad_fn=<ReluBackward0>)
print(output_int8)
# tensor([[[[0.3886, 0.0000],
#           [0.2475, 0.0000]]]])