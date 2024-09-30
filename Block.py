import torch

from AttentionHead import AttentionHead
from FeedForward import FeedForward
from MultipleHeads import MultipleHeads

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Block(torch.nn.Module):
    def __init__(self, EmbeddingSize):
        super(Block, self).__init__()
        self.normLayerForFeedForward = torch.nn.LayerNorm(EmbeddingSize, dtype=torch.float32)
        self.normLayerForHeadClusters = torch.nn.LayerNorm(EmbeddingSize, dtype=torch.float32)
        self.headsCluster = torch.nn.Sequential(*[MultipleHeads(EmbeddingSize) for _ in range(5)])
        self.feedForward = FeedForward(EmbeddingSize)

        self.normLayerForFeedForward.to(device)
        self.normLayerForHeadClusters.to(device)
        self.headsCluster.to(device)
        self.feedForward.to(device)

    def forward(self, x):
        x = x + self.headsCluster(self.normLayerForHeadClusters(x))
        return x + self.feedForward(self.normLayerForFeedForward(x))
