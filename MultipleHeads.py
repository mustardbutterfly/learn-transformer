import torch.nn

from AttentionHead import AttentionHead


class MultipleHeads(torch.nn.Module):
    def __init__(self, embedding_size):
        super(MultipleHeads, self).__init__()
        self.heads = [AttentionHead(embedding_size) for _ in range(4)]

    def forward(self, x):  # 8 x 4 x 64
        output = []
        for head in self.heads:
            output.append(head(x))
        return torch.cat(output, dim=2)
