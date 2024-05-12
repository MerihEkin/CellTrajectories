import os
import sys

import torch 
import torch.nn as nn
import torch.optim as optim

from dataclasses import dataclass

from torch.utils.data import DataLoader

sys.path.append(".")

from data_loader import TrackDataset, TaskType

"""
This code is adapted from 
https://github.com/chrisvdweth/ml-toolkit/blob/master/pytorch/models/text/autoencoder/textrnnae.py
"""

class RnnType:
    GRU = 1
    LSTM = 2

class ActivationFunction:
    RELU = 1
    TANH = 2
    SIGMOID = 3

@dataclass
class Params:
    rnn_type = RnnType.GRU
    activation_function = ActivationFunction.RELU
    seq_len = 90
    num_layers = 2
    num_features = 18
    hidden_size = 4
    latent_dim = 4
    use_vae = False
    bias = False
    bidirectional = False
    dropout = 0.1
    learning_rate = 0.01

rnn_classes = {
    RnnType.GRU: nn.GRU,
    RnnType.LSTM: nn.LSTM
}

class Encoder(nn.Module):
    def __init__(self, params : Params) -> None:
        super().__init__()
        self.params = params
        if params.rnn_type in rnn_classes:
            rnn_class = rnn_classes[params.rnn_type]
            self.nn = rnn_class(
                input_size=params.num_features,
                hidden_size=params.hidden_size,
                num_layers=params.num_layers,
                bias=params.bias,
                bidirectional=params.bidirectional,
                dropout=params.dropout,
                batch_first=True
            )
        else:
            raise ValueError("Unsupported RNN type")
        

    def forward(self, x):
        if self.params.rnn_type == RnnType.GRU:
             _, h_n = self.nn(x)
        elif self.params.rnn_type == RnnType.LSTM:
            _, (h_n, c_n) = self.nn(x)
        out = h_n[-1]
        return out
        

class Decoder(nn.Module):
    def __init__(self, params : Params) -> None:
        super().__init__()
        self.params = params
        if params.rnn_type in rnn_classes:
            rnn_class = rnn_classes[params.rnn_type]
            self.nn = rnn_class(
                input_size=params.hidden_size,
                hidden_size=params.num_features,
                num_layers=params.num_layers,
                bias=params.bias,
                bidirectional=params.bidirectional,
                dropout=params.dropout,
                batch_first=True
            )
        else:
            raise ValueError("Unsupported RNN type")
        self.fc = nn.Linear(in_features=params.num_features, out_features=params.num_features)

    def forward(self, x):
        """
        Our aim in this autoencode structure is to reconstruct a point of interest
        using the neighbourhood of that point.
        So we do not have to construct the whole thing, as long as we can represent 
        a point of interest meaningfully.
        """
        # x = x.unsqueeze(1).repeat(1, self.params.seq_len, 1)
        out_rnn, _ = self.nn(x)
        out = self.fc(out_rnn)
        return out
    

class RNNAE(nn.Module):
    def __init__(self, params : Params):
        super().__init__()

        self.encoder = Encoder(params=params)
        self.decoder = Decoder(params=params)


    def forward(self, x):
        latent_representation = self.encoder(x)
        output = self.decoder(latent_representation)
        return output
    

def train(model, dataset, batch_size, epochs, lr):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass + backward pass + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0



if __name__ == '__main__':
    params = Params()
    params.activation_function = ActivationFunction.RELU
    params.seq_len = 30
    params.dropout = 0.1
    params.rnn_type = RnnType.LSTM
    params.num_layers = 4
    params.learning_rate = 0.001
    params.hidden_size = 4

    dir = os.path.join(os.getcwd(), 'data', 'tensors')
    files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.pt')]
    task = TaskType.Reconstruction
    dataset = TrackDataset(files=files, seq_len=params.seq_len, task=task)
    model = RNNAE(
        params=params
    )
    train(model=model, dataset=dataset, batch_size=16, epochs=2, lr=params.learning_rate)
