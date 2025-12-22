import torch

def precompute_freqs_cis(dim,end,theta=50000.0,device="cpu"):#生成位置向量列表,设置单位角度
    freqs = 1.0/(theta ** (torch.arange(0,dim,2,device=device)[:(dim//2)].float()/dim))#对50000按步长开方的倒数,每个向量两两一组(dim=32,32/2=16),公式中的dim可以不分两组不取步长2,dim改为16与查询向量形状匹配
    t = torch.arange(end,device=device)#对应token数创建对应数量的旋转编码
    freqs = torch.outer(t,freqs).float()#求外积，20个token,16组向量，按不同token的下标t编码角度成倍偏转
    freqs_cis=torch.polar(torch.ones_like(freqs),freqs)#创建模为1的极坐标,横轴为实数1,纵轴为虚部1i,第一个参数为模长(张量),第二个参数为δ(张量),转极坐标后为sinδ+cosδi
    return freqs_cis

def apply_rotary_emb(xq,freqs_cis):#将词向量叠加位置向量,点乘,向量两两转到复平面
    assert xq.shape[-1] % 2 == 0, "The last dimension of xq must be even."
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1],-1,2))#查询向量分成两组
    freqs_cis = reshape_for_broadcast(freqs_cis,xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)#在复数域计算旋转，方便词向量与词向量做余弦，位置与位置做余弦
    return xq_out.type_as(xq)

def reshape_for_broadcast(freqs_cis,x):#形状匹配
    return freqs_cis[:x.shape[2]].unsqueeze(0).unsqueeze(0)

if __name__ == '__main__':
    freqs_cis=precompute_freqs_cis(32,20,device="cuda:0")
    xq=torch.randn(1,1,20,32).cuda()#NHSV
    xq_=apply_rotary_emb(xq,freqs_cis)
