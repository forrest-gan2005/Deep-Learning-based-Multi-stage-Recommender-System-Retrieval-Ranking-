import torch
import torch.nn as nn

class DeepFM(nn.Module):
    def __init__(self, field_dims, embed_dim=16):
        super(DeepFM, self).__init__()
        self.linear = nn.ModuleList([nn.Embedding(dim, 1) for dim in field_dims])
        self.Embedding = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in field_dims])
        self.bias = nn.Parameter(torch.zeros((1,)))
        
        input_dim = len(field_dims) * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        linear_out = sum([self.linear[i](x[:, i]) for i in range (x.shape[1])]).squeeze(1)
        
        embeds = torch.stack([self.Embedding[i](x[:, i]) for i in range(x.shape[1])], dim=1)
        
        square_of_sum = torch.sum(embeds, dim=1)**2
        sum_of_square = torch.sum(embeds**2, dim=1)
        fm_out = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1)
        
        deep_out = self.mlp(embeds.view(x.size(0), -1)).squeeze(1)
        
        return torch.sigmoid(linear_out + fm_out + deep_out + self.bias)