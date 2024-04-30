import torch
import torch.nn as nn

"""
Initial script of implementing LSTMAE for trajectory embedding.

See : https://ieeexplore.ieee.org/document/7966345 
Trajectory clustering via deep representation learning 
Uses sequence to sequence autoencoder to encode spatio-temporal 
data for clustering. In public transport trajectories they achieve
good results for distinguishing between circular and directional 
trajectories. 

Could be useful for us but our trajectories are a bit more complex.

A Denoising Hybrid Model for Anomaly Detection in
Trajectory Sequences

https://mever.gr/publications/A%20Denoising%20Hybrid%20Model%20for%20Anomaly%20Detection%20in%20Trajectory%20Sequences.pdf

"""

class Encoder(nn.Module):
    def __init__(self, seq_len=100, n_features=18, n_hidden=64, n_embedding=16, n_encoding=16):
        super(Encoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_embedding = n_embedding
        self.n_encoding = n_encoding

        self.rnn1 = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.n_embedding,
            num_layers=1,
            batch_first=True,
        )

        self.rnn2 = nn.LSTM(
            input_size=self.n_embedding,
            hidden_size=self.n_encoding,
            num_layers=1,
            batch_first=True,
        )

        self.relu = nn.ReLU()       # Not very common to use Relu with LSTM look more into this

    def forward(self, x):
        x, (_, _) = self.rnn1(x)    # TODO : forward pass may be incorrect
        x = self.relu(x)
        _, (hidden_state, _) = self.rnn2(x)
        output = self.relu(hidden_state[-1])
        return output

class Decoder(nn.Module):
    def __init__(self, seq_len=100, n_features=18, n_hidden=64, n_embedding=16, n_encoding=16):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_embedding = n_embedding
        self.n_encoding = n_encoding

        self.rnn1 = nn.LSTM(        # TODO : Decoder structure not be incorrect
            input_size=self.n_encoding,
            hidden_size=self.n_embedding,
            num_layers=1,
            batch_first=True,
        )

        self.rnn2 = nn.LSTM(
            input_size=self.n_embedding,
            hidden_size=self.n_features,
            num_layers=1,
            batch_first=True,
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x, (_, _) = self.rnn1(x)
        x = self.relu(x)
        _, (hidden_state, _) = self.rnn2(x)
        output = self.relu(hidden_state[-1])
        return output

class Seq2SeqAE(nn.Module):
    def __init__(self):
        super(Seq2SeqAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train(self, data, epochs=1000, lr=1e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            for x in data:
                optimizer.zero_grad()
                output = self(x)
                loss = criterion(output, x)
                loss.backward()
                optimizer.step()
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item()}")


    def load_model(self, path):
        """
        TODO : Implement
        encoder, decoder seperate files
        """
        pass


    def save_model(self, path):
        """
        TODO : Implement
        encoder, decoder seperate files
        """
        pass


### Test with random data until we have more data
if __name__ == '__main__':
    seq_len = 100
    n_features = 18
    batch_size = 20
    num_batches = 10
    data = [torch.randn(batch_size, seq_len, n_features) for _ in range(num_batches)]
    
    AE = Seq2SeqAE()
    AE.train(
        data=data,
        epochs=5,
    )
