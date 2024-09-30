import torch.nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FeedForward(torch.nn.Module):
    def __init__(self, vocab_size):
        super(FeedForward, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(vocab_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, vocab_size),
            torch.nn.Dropout(.2)
        )

        self.layers.to(device)

    def forward(self, x):
        return self.layers(x)
