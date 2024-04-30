import torch
import torch.nn as nn


"""

In paper : A Denoising Hybrid Model for Anomaly Detection in Trajectory Sequences

https://mever.gr/publications/A%20Denoising%20Hybrid%20Model%20for%20Anomaly%20Detection%20in%20Trajectory%20Sequences.pdf

They define two different architectures;

1. where both the Encoder and Decoder network is comprised of LSTM layers and the Decoder uses the Encoder output as the 
input. The LSTMAE implements this simple structure. 

2. in the second architecture the Decoder uses the hidden states of the Encoder output to reconstruct the input using 
teacher learning. This will be implemented in the Seq2Seq class.

"""


class Encoder(nn.Module):
    def __init__(self, seq_len = 100, n_features = 18, n_hidden = 4, n_layers = 1):
        super().__init__()
        self.model = nn.LSTM(
            input_size = n_features,
            hidden_size = n_hidden,
            num_layers= n_layers,
        )
    
    def forward(self, x):
        out, _ = self.model(x)
        return out

class Decoder(nn.Module):
    def __init__(self, seq_len = 100, n_features = 18, n_hidden = 4, n_layers = 1):
        super().__init__()
        self.model = nn.LSTM(
            input_size = n_hidden,
            hidden_size = n_features,
            num_layers= n_layers,
        )
    
    def forward(self, x):
        out, _ = self.model(x)
        return x


class LSTMAE(nn.Module):
    def __init__(self, seq_len = 100, n_features = 18, n_hidden = 4, n_layers = 1):
        super().__init__()
        
        self.encoder = Encoder(
            seq_len=seq_len,
            n_features=n_features,
            n_hidden=n_hidden,
            n_layers=n_layers,
        )

        self.decoder = Decoder(
            seq_len=seq_len,
            n_features=n_features,
            n_hidden=n_hidden,
            n_layers=n_layers,
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruct = self.decoder(latent)
        return reconstruct

