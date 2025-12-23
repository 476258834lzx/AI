import torch
import torch.nn as nn
import torch.nn.functional as F

class Exper(nn.Module):
    def __init__(self,input_size,output_size):
        super(Exper,self).__init__()
        self.fc1 = nn.Linear(input_size,output_size)

    def forward(self,x):
        return F.relu(self.fc1(x))

class Gate(nn.Module):
    def __init__(self,input_size,num_expers):
        super(Gate,self).__init__()
        self.fc = nn.Linear(input_size,num_expers)

    def forward(self,x):
        return self.fc(x)

class HardMoe(nn.Module):
    def __init__(self,input_size,output_size,num_expers,top_k):
        super(HardMoe,self).__init__()
        self.expers = nn.ModuleList([Exper(input_size,output_size) for _ in range(num_expers)])
        self.gate= Gate(input_size,num_expers)
        self.top_k = top_k

    def forward(self,x):
        N,S,V = x.shape[:]
        gate_output = self.gate(x)

        top_value,top_index = torch.topk(gate_output,self.top_k,dim=-1)
        expert_outputs=torch.zeros(N,S,self.top_k,output_size).to(x.device)
        for t in range(S):
            for i in range(N):
                for j in range(self.top_k):
                    expert_index=top_index[i,t,j]
                    expert_outputs[i,t,j]=self.expers[expert_index].forward(x[i,t])
        output=expert_outputs.mean(dim=2)
        return output


input_size=10
output_size=5
num_expers=4
top_k=2
model=HardMoe(input_size,output_size,num_expers,top_k)

N=32
S=15
V=input_size

x=torch.randn(N,S,V)
output=model(x)
print(output.shape)