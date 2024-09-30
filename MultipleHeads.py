import torch

from AttentionHead import AttentionHead


class MultipleHeads(torch.nn.Module):
    def __init__(self, embedding_size):
        super(MultipleHeads, self).__init__()
        self.heads = torch.nn.ModuleList([AttentionHead(embedding_size) for _ in range(4)])
        self.linear = torch.nn.Linear(embedding_size, embedding_size)
        self.dropout = torch.nn.Dropout(.2)

    def forward(self, x):  # 8 x 4 x 64
        x = torch.cat([head(x) for head in self.heads], dim=2)
        return self.dropout(self.linear(x))
