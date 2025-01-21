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
    
class RNN_Block(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.in2hidden = layer_init(nn.Linear(embed_size + hidden_size, hidden_size), n_linearity="tanh")
        self.bacth_norm = nn.LayerNorm(hidden_size)
        self.in2out = layer_init(nn.Linear(embed_size + hidden_size, vocab_size), n_linearity="linear",weight_fill=0.01)

    def forward(self, x, a):
        # print(x.shape, a.shape)
        combined = torch.cat((x, a), dim=-1)
        a_hat = self.bacth_norm(F.tanh(self.in2hidden(combined)))
        y = self.in2out(combined)

        return y, a_hat
    
    def init_hidden(self, batch_size, device="cpu"):
        return nn.init.kaiming_uniform_(torch.empty(batch_size, self.hidden_size, device=device))
    

class LSTM_Block(nn.Module):

    def __init__(self, embed_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.forget_layer = layer_init(nn.Linear(embed_size+hidden_size, hidden_size), n_linearity="sigmoid")
        self.input_layer = layer_init(nn.Linear(embed_size+hidden_size, hidden_size), n_linearity="sigmoid")
        self.hidden_layer = layer_init(nn.Linear(embed_size+hidden_size, hidden_size), n_linearity="tanh")
        self.output_layer = layer_init(nn.Linear(embed_size+hidden_size, hidden_size), n_linearity="sigmoid")
        self.hid2out = layer_init(nn.Linear(hidden_size, hidden_size), n_linearity="tanh")

    def forward(self, x, short_term, long_term):
        combined = torch.cat((x, short_term), dim=-1)
        f = F.sigmoid(self.forget_layer(combined))
        i = F.sigmoid(self.input_layer(combined))
        C = F.tanh(self.hidden_layer(combined))
        C_new = torch.mul(f, long_term) + torch.mul(i, C)
        o = F.sigmoid(self.output_layer(combined))
        C_out = F.tanh(self.hid2out(C_new))
        output = torch.mul(C_out, o)
        return output, C_new
    
    def init_hidden(self, batch_size, device="cpu"):
        return nn.init.kaiming_uniform_(torch.empty(batch_size, self.hidden_size, device=device))



class SimpleRNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size, device="cpu"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.block = RNN_Block(embed_size, hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.device = device
    
    def forward(self, x):
        x = self.embedding(x)
        batch_size, sequence_length, _ = x.shape
        hidden_state = self.block.init_hidden(batch_size, self.device)
        out, hid = [], []
        for t in range(sequence_length):
            output, hidden_state = self.block(x[:,t,:], hidden_state)
            out.append(output)
            hid.append(hidden_state)
        return out, hid
        

class LSTM(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size, device="cpu"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.block = LSTM_Block(embed_size, hidden_size)
        self.fc = layer_init(nn.Linear(hidden_size, vocab_size), n_linearity="linear")
        self.hidden_size = hidden_size
        self.device = device
    
    def forward(self, x):
        x = self.embedding(x)
        batch_size, sequence_length, _ = x.shape
        long_term = self.block.init_hidden(batch_size, self.device)
        output = long_term
        out, hid = [], []
        for t in range(sequence_length):
            output, long_term = self.block(x[:,t,:], output, long_term)
            out.append(self.fc(output))
            hid.append(long_term)
        return out, hid
        