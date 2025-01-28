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
        

# I am adding the Encoder and Decoder block. but for text generation we only need a Decoder
# Hence, cross attention is also not used in the decoder block, but I have added the code

class Positional_Embedding(nn.Module):

    def __init__(self, seq_len, embed_size, n=10000):
        super().__init__()
        P = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, embed_size, 2).float() * (torch.log(torch.tensor(n)) / embed_size))  # (d//2,)
        self.P = torch.zeros((seq_len, embed_size))  # Embedding matrix
        self.P[:, 0::2] = torch.sin(P * div_term)  
        self.P[:, 1::2] = torch.cos(P * div_term) 
        self.P = nn.Parameter(self.P)
        print(self.P.shape)

    def forward(self,  x):
        return self.P[x]  
    
class MultiHeadAttention(nn.Module):

    def __init__(self, n_embd, head_size, num_of_heads, block_size, decode=False):
        super().__init__()
        self.decode = decode
        self.head_size = head_size
        self.num_of_heads = num_of_heads
        self.key = nn.Linear(n_embd, head_size*num_of_heads, bias=False)
        self.value = nn.Linear(n_embd, head_size*num_of_heads, bias=False)
        self.query = nn.Linear(n_embd, head_size*num_of_heads, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):

        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.view(B, T, self.num_of_heads, self.head_size)  # (B, T, num_heads. C headsize)
        q = q.view(B, T, self.num_of_heads, self.head_size) 
        v = v.view(B, T, self.num_of_heads, self.head_size) 

        k = k.transpose(1, 2) # (B, Num_heads, T, C)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        wei = q @ k.transpose(-2, -1) * C ** -0.5 # (B, N, T, C) @ (B, N, C ,T) -> (B, N, T, T)
        if self.decode:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v # (B, N, T, T) @ (B, N, T, C) -> (B, N, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_of_heads * self.head_size)
        return out
    
class CrossHeadAttention(nn.Module):

    def __init__(self, n_embd, head_size, num_of_heads):
        super().__init__()
        self.head_size = head_size
        self.num_of_heads = num_of_heads
        self.key = nn.Linear(n_embd, head_size * num_of_heads, bias=False)
        self.query = nn.Linear(n_embd, head_size * num_of_heads, bias=False)
        self.value = nn.Linear(n_embd, head_size * num_of_heads, bias=False)

    def forward(self, x_enc, x):
        B_enc, T_enc, C_enc = x_enc.shape
        B, T, C = x.shape
        
        k = self.key(x_enc)
        v = self.value(x_enc)
        q = self.query(x)

        k = k.view(B_enc, T_enc, self.num_of_heads, self.head_size)
        v = v.view(B_enc, T_enc, self.num_of_heads, self.head_size)
        q = q.view(B, T, self.num_of_heads, self.head_size)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_of_heads, * self.head_size)

        return out
    
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd * 4),
            nn.ReLU()
        )
        self.proj = nn.Linear(n_embd * 4, n_embd)

    def forward(self, x):
        x = self.net(x)
        x = self.proj(x)

        return x
    
class Encoder_Block(nn.Module):

    def __init__(self, head_size, n_embd, block_size, num_of_heads):
        super().__init__()
        self.attention = MultiHeadAttention(n_embd, head_size//num_of_heads, num_of_heads, block_size)
        self.proj1 = nn.Linear(head_size, n_embd)
        self.forw = FeedForward(n_embd)
        self.lr1 = nn.LayerNorm(n_embd)
        self.lr2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.proj1(self.attention(self.lr1(x)))
        x = x + self.forw(self.lr2(x))

        return x
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, head_size, num_of_heads):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = Positional_Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Encoder_Block(head_size, n_embd, block_size, num_of_heads) for _ in range(6)])
        self.lr = nn.LayerNorm(n_embd)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T))
        x = tok + pos
        x = self.blocks(x)
        logits = self.lr(x)

        return logits
        
class Decoder_Block(nn.Module):

    def __init__(self, head_size, n_embd, block_size, num_of_heads):
        super().__init__()
        self.attention = MultiHeadAttention(n_embd, head_size//num_of_heads, num_of_heads, block_size, decode=True)
        self.proj1 = nn.Linear(head_size, n_embd)
        self.lr1 = nn.LayerNorm(n_embd)
        self.forw = FeedForward(n_embd)
        self.lr2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.proj1(self.attention(self.lr1(x)))
        x = x + self.forw(self.lr2(x))

        return x
    
class Decoder(nn.Module):

    def __init__(self, vocab_size, block_size, n_embd, head_size, num_of_heads):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = Positional_Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Decoder_Block(head_size, n_embd, block_size, num_of_heads) for _ in range(6)])
        self.proj = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, target=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T))
        x = tok + pos
        x = self.blocks(x)
        logits = self.proj(x)

        if target == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
        return logits, loss

class Transformer(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.decoder = Decoder(vocab_size, 256, 32, 128, 4)

    def forward(self, x, y=None):
        logits, loss = self.decoder(x, y)
        return logits, loss