import torch
from torch import nn

class CBOW(nn.Module):
    def __init__(self):
        super().__init__()

        # self.embedding = nn.Parameter(torch.randn(10000,128))
        self.embedding = nn.Embedding(num_embeddings=10000, embedding_dim=100)

        self.layer = nn.Sequential(
            nn.Linear(in_features=4*100, out_features=100,bias=False)
        )

        self.loss_function = nn.MSELoss()

    def forward(self, token1,token2,word):
        vector1 = self.embedding(token1)
        vector2 = self.embedding(token2)

        vector = torch.cat((vector1,vector2),dim=1).reshape(-1,4*100)
        score = self.layer(vector)

        return self.loss_function(score,word)


