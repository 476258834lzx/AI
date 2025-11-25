import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch import nn

class GNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=GCNConv(3,16)
        self.conv2=GCNConv(16,32)
        self.fc=nn.Linear(32,6)


    def forward(self,data):
        x,edge_index=data.x,data.edge_index

        x=self.conv1(x,edge_index)
        x=F.relu(x)
        x=self.conv2(x,edge_index)
        x=torch.max(x,dim=0)#输出层不能使用有序网络结构，图节点顺序可打乱，不能使用自适应池化，可使用mean、maxpool、transformers
        x=self.fc(x)

        return F.log_softmax(x)