#加载预训练模型
import torch
from torch import nn
from transformers import BertModel

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained=BertModel.from_pretrained("bert-base-chinese").to(DEVICE)

# print(pretrained)
#定义head网络(处理主干网络提取出的特征)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer=nn.Linear(768,2)
    def forward(self,input_ids,attention_mask,token_type_ids):
        with torch.no_grad():#与训练模型不参与训练
            out=pretrained(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        out=self.layer(out.last_hidden_state[:,0])
        out=out.softmax(dim=1)
        return out