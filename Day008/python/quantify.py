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
#qconfig_spec 赋值为一个 set，比如：{nn.LSTM, nn.Linear}，意思是指定当前模型中的哪些 layer 要被 dynamic quantization；
#qconfig_spec 赋值为一个 dict，key 为 submodule 的 name 或 type，value 为 QConfigDynamic 实例（其包含了特定的 Observer，比如 MinMaxObserver、MovingAverageMinMaxObserver、PerChannelMinMaxObserver、MovingAveragePerChannelMinMaxObserver、HistogramObserver）
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
# """
# 权重和激活都会被量化
# """
#
# import torch
# from torch import nn
#
#
# # 定义一个浮点模型，其中一些层可以被静态量化
# class F32Model(torch.nn.Module):
#     def __init__(self):
#         super(F32Model, self).__init__()
#         self.quant = torch.quantization.QuantStub()  # QuantStub: 转换张量从浮点到量化
#         self.conv = nn.Conv2d(1, 1, 1)
#         self.fc = nn.Linear(2, 2, bias=False)
#         self.relu = nn.ReLU()
#         self.dequant = torch.quantization.DeQuantStub()  # DeQuantStub: 将量化张量转换为浮点
#
#     def forward(self, x):
#         x = self.quant(x)  # 手动指定张量: 从浮点转换为量化
#         x = self.conv(x)
#         x = self.fc(x)
#         x = self.relu(x)
#         x = self.dequant(x)  # 手动指定张量: 从量化转换到浮点
#         return x
#
#
# model_fp32 = F32Model()
# model_fp32.eval()  # 模型必须设置为eval模式，静态量化逻辑才能工作
#
# # 1、如果要部署在ARM上；果要部署在x86 server上 ‘fbgemm’
# model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
#
# # 2、在适用的情况下，将一些层进行融合，可以加速
# # 常见的融合包括在：DEFAULT_OP_LIST_TO_FUSER_METHOD
# model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['fc', 'relu']])
#
# # 3、准备模型，插入observers，观察 activation 和 weight
# model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)
#
# # 4、代表性数据集，获取数据的分布特点，来更好的计算activation的 scale 和 zp
# input_fp32 = torch.randn(1, 1, 2, 2)  # (batch_size, channel, W, H)
# model_fp32_prepared(input_fp32)
#
# # 5、量化模型
# model_int8 = torch.quantization.convert(model_fp32_prepared)
#
# # 运行模型，相关计算将在int8中进行
# output_fp32 = model_fp32(input_fp32)
# output_int8 = model_int8(input_fp32)
# print(output_fp32)
# # tensor([[[[0.6315, 0.0000],
# #           [0.2466, 0.0000]]]], grad_fn=<ReluBackward0>)
# print(output_int8)
# # tensor([[[[0.3886, 0.0000],
# #           [0.2475, 0.0000]]]])

#量化感知训练cpu
# QAT follows the same steps as PTQ, with the exception of the training loop before you actually convert the model to its quantized version
# QAT遵循与PTQ相同的步骤，除了在实际将模型转换为量化版本之前进行训练循环
''''''
'''量化感知训练步骤：
step1.搭建模型
step2.融合（可选步骤）
step3.插入stubs（1和3可合在一起）
step4.准备（主要是选择架构）
step5.训练
step6.模型转换
'''
import torch
from torch import nn
platform="x86"
backend = 'fbgemm' if platform=="x86" else 'qnnpack'  # running on a x86 CPU. Use "qnnpack" if running on ARM.

'''step1.搭建模型build model'''
m = nn.Sequential(
     nn.Conv2d(2,64,8),
     nn.ReLU(),
     nn.Conv2d(64, 128, 8),
     nn.ReLU(),
)

"""step2.融合Fuse（可选步骤）"""
torch.quantization.fuse_modules(m, ['0','1'], inplace=True) # fuse first Conv-ReLU pair
torch.quantization.fuse_modules(m, ['2','3'], inplace=True) # fuse second Conv-ReLU pair

"""step3.插入stubs于模型，Insert stubs"""
m = nn.Sequential(torch.quantization.QuantStub(),
                  *m,
                  torch.quantization.DeQuantStub())

"""step4.准备Prepare"""
#也可自定义量化参数
# quantization_config = torch.quantization.default_qconfig
# quantization_config = torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8), weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
m.train()
m.qconfig = torch.quantization.get_default_qconfig(backend)
torch.quantization.prepare_qat(m, inplace=True)

"""step5.训练Training Loop"""
n_epochs = 10
opt = torch.optim.SGD(m.parameters(), lr=0.1)
loss_fn = lambda out, tgt: torch.pow(tgt-out, 2).mean()
for epoch in range(n_epochs):
  x = torch.rand(10,2,24,24)
  out = m(x)
  loss = loss_fn(out, torch.rand_like(out))
  opt.zero_grad()
  loss.backward()
  opt.step()
  print(loss)

"""step6.模型转换Convert"""
m.eval()
torch.quantization.convert(m, inplace=True)
# 指定与qconfig相同的backend，在推理时使用正确的算子
# 目前Pytorch的int8算子只支持CPU推理,需确保输入和模型都在CPU侧
torch.backends.quantized.engine = backend
