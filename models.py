import torch
import torch.nn as nn
import torch.nn.functional as F

def layer_init(layer: nn.Linear, n_linearity="relu", weight_fill=None):
    torch.nn.init.kaiming_uniform_(layer.weight, a=0, mode="fan_in", nonlinearity=n_linearity)
    if weight_fill:
        layer.weight.data.fill_(weight_fill) # this is usually for the last linear layer
    return layer

class N_Gram(nn.Module):

    def __init__(self, vocab_size, context_size, embd_size, hidden_sizes: list):
        super().__init__()
        assert len(hidden_sizes) >= 1, "at least one hidden layer is needed"
        self.embedding = nn.Embedding(vocab_size, embd_size)
        layers = []
        input_size = context_size * embd_size
        for h_size in hidden_sizes:
            layers.append(layer_init(nn.Linear(input_size, h_size)))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(num_features=h_size))
            input_size = h_size
        layers.append(layer_init(nn.Linear(input_size, vocab_size), n_linearity="linear", weight_fill=True))

        self.net = nn.Sequential(*layers)


    def forward(self, x):
        x_embd = self.embedding(x).view(x.shape[0], -1)
        x = self.net(x_embd)
        return x