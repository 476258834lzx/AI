from transformers import BertModel
import torch
from torch import nn

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained=BertModel.from_pretrained("yechen/bert-large-chinese").to(DEVICE)

print(pretrained)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer=nn.Linear(1024,8)
    def forward(self,input_ids,attention_mask,token_type_ids):
        with torch.no_grad():
            out=pretrained(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        out=self.layer(out.last_hidden_state[:,0])
        out=out.softmax(dim=1)
        return out