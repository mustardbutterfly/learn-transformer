import torch.nn

head_size = 16  # 64/ 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AttentionHead(torch.nn.Module):

    def __init__(self, vocab_size):  # 64
        super(AttentionHead, self).__init__()

        self.keys = torch.nn.Linear(vocab_size, head_size)
        self.values = torch.nn.Linear(vocab_size, head_size)
        self.query = torch.nn.Linear(vocab_size, head_size)

        self.keys.to(device)
        self.values.to(device)
        self.query.to(device)

        self.register_buffer('tril', torch.tril(torch.ones(8, 8)))

    def forward(self, x):  # 4 x 8 x 64
        v = self.values(x)
        q = self.query(x)
        k = self.keys(x)  # 4 X 8 X 16

        tensorForQueryAndKey = q @ k.transpose(-1, -2)  # 4 8 8
        trimmedQueryAndKey = torch.tril(tensorForQueryAndKey)
        trimmedQueryAndKey.masked_fill(self.tril[:head_size, :head_size] == 0, float('-inf'))
        tensorForQueryAndKeyScaledDown = trimmedQueryAndKey * head_size ** -0.5
        tensorForQueryAndKeyNormalized = torch.softmax(tensorForQueryAndKeyScaledDown, dim=-1)
        dotProductOfAttention = tensorForQueryAndKeyNormalized @ v  # 4 8 16

        return dotProductOfAttention
