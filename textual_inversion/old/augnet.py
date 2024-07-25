from torch import nn
class Augmenter(nn.Module):
    def __init__(self, embed_dim, width_fac=4):
        super(Augmenter, self).__init__()
        self.W_ff1 = nn.Linear(embed_dim, width_fac * embed_dim)
        self.W_ff2 = nn.Linear(embed_dim * width_fac, embed_dim//2)
        self.relu = nn.ReLU()
        self.dtype=None
    def forward(self, X):
        # Simple Feedforward network that projects into a higher space (by width_fac) and back to embed_dim
        X = self.W_ff1(X)
        X = self.relu(X)
        return self.W_ff2(X)