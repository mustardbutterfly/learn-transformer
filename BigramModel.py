import torch

from AttentionHead import AttentionHead
from FeedForward import FeedForward
from MultipleHeads import MultipleHeads

EmbeddingSize = 65
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BigramModel(torch.nn.Module):
    def __init__(self, EmbeddingSize):
        super(BigramModel, self).__init__()
        self.normLayer = torch.nn.LayerNorm(EmbeddingSize, dtype=torch.float32)
        self.headsCluster = torch.nn.ModuleList([MultipleHeads(EmbeddingSize) for _ in range(4)])
        self.feedForward = FeedForward(EmbeddingSize)

        self.normLayer.to(device)
        self.headsCluster.to(device)
        self.feedForward.to(device)

    def forward(self, x):
        outputs = torch.zeros(x.shape, dtype=torch.float32)
        normalizedInput = self.normLayer(x)  # 4 8 64
        for head in self.headsCluster:
            outputs += head(normalizedInput)
        return x + .9 * outputs

    def parameters(self, **kwargs):
        return self.normLayer.parameters()
