import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

def set_up_causal_mask(seq_len):
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask.requires_grad = False
    return mask
    

class LatentEncoder(nn.Module):
    def __init__(self, in_size, seq_len =70, hidden_size=512, dropout=0.1):
        super(LatentEncoder, self).__init__()
        self.in_size = in_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.featurizer = nn.Sequential(
            nn.Linear(in_size, 1024),
            nn.GELU(),
            nn.Linear(1024, hidden_size),
            nn.GELU(),
        )
    def forward(self, x):
        x = self.featurizer(x)
        return x
    
     
class PositionalEncodings(nn.Module):
    """Attention is All You Need positional encoding layer"""

    def __init__(self, seq_len, d_model, p_dropout):
        """Initializes the layer."""
        super(PositionalEncodings, self).__init__()
        token_positions = torch.arange(start=0, end=seq_len).view(-1, 1)
        dim_positions = torch.arange(start=0, end=d_model).view(1, -1)
        angles = token_positions / (10000 ** ((2 * dim_positions) / d_model))

        encodings = torch.zeros(1, seq_len, d_model)
        encodings[0, :, ::2] = torch.cos(angles[:, ::2])
        encodings[0, :, 1::2] = torch.sin(angles[:, 1::2])
        encodings.requires_grad = False
        self.register_buffer("positional_encodings", encodings)

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        """Performs forward pass of the module."""
        x = x + self.positional_encodings[:,:x.shape[1],:]
        x = self.dropout(x)
        return x
    
    
class LatentToMol(nn.Module):
    def __init__(self, 
                 in_size, 
                 hidden_size,  
                 n_layers, 
                 seq_len, 
                 vocab, 
                 n_heads = 16, 
                 dropout=0.1):
        super(LatentToMol, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.vocab = vocab
        self.embed = nn.Embedding(len(vocab), hidden_size)
        self.pe = PositionalEncodings(d_model=hidden_size, p_dropout=dropout, seq_len=seq_len)
        # self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, hidden_size))
        
        self.classifier = nn.Linear(hidden_size, len(vocab))
        

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size,
                                                               nhead=n_heads,
                                                               dropout=dropout,
                                                               batch_first=True,
                                                               norm_first=True,
                                                               activation="gelu")
        self.trfmencoder = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer,
                                                 num_layers=n_layers,
                                                 norm=nn.LayerNorm(hidden_size)
                                                 )

        self.specencoder = LatentEncoder(in_size=in_size, 
                                       hidden_size=hidden_size,
                                       dropout=dropout)
        
        self.ln_f = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(0.1)
    def forward(self, spec, smi, tgt_mask=None, tgt_padding_mask=None):
        spec = self.specencoder(spec)
        spec = spec.unsqueeze(1)
        # print(smi.shape)
        # print(smi.dtype)
        smi = self.embed(smi) 
        smi = smi[:,1:,:]
        smi = self.pe(smi)
        
        x = torch.cat([spec, smi], dim=1).to(smi.device)
        # smi = self.pe(smi) # [batch, seqlen, hidden]
 
        mask = set_up_causal_mask(x.shape[1]).to(smi.device)

        x = self.trfmencoder(
            src=x,
            mask = mask
        )
        x = self.ln_f(x)
        out = self.classifier(x)
        # out = F.log_softmax(out, 2)
        return  out
