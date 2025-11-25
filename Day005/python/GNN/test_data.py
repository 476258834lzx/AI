import torch
from torch_geometric.data import Data

edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]],dtype=torch.long)
# points=torch.tensor([[-1],[0],[1]],dtype=torch.float)
points=torch.tensor([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]],dtype=torch.float)

data=Data(x=points,edge_index=edge_index)#图模型采用稀疏矩阵计算，定义数据可对边添加.contiguous使得内存相连

print(data)

#查看属性
print(data.num_nodes)
print(data.num_edges)
print(data.num_node_features)
print(data.has_self_loops())#是否有环
print(data.is_directed())#是否是有向图

##自带预训练数据
# from torch_geometric.datasets import TUDataset,Planetoid
# dataset=TUDataset(root="./data/",name='ENZYMES')#酶结构数据

# dataset=Planetoid(root="./",name='ENZYMES')
