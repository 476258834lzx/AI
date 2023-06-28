import torch

data=torch.tensor([[3,4],[5,6],[7,8],[9,8],[6,5]])
label=torch.tensor([0,0,1,0,1])

center=torch.tensor([[1,1],[2,2]])
center_exp=center.index_select(dim=0,index=label.long())

count=torch.histc(label.float(),bins=2,min=0,max=1)
count_exp=count.index_select(dim=0,index=label.long())

print(count_exp)
center_loss=torch.sum(torch.div(torch.sqrt(torch.sum(torch.pow(data-center_exp,2),dim=1)),count_exp))
print(center_loss)