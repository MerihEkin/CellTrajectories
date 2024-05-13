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
    use_vae = False
    bias = False
    bidirectional = False
    dropout = 0.1
    learning_rate = 0.01
    latent_dim = 4


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
                batch_first=False
            )
        else:
            raise ValueError("Unsupported RNN type")
        
        self.seq_len = params.seq_len


    def forward(self, x):
        if self.params.rnn_type == RnnType.GRU:
            outputs, hidden = self.nn(x)
        elif self.params.rnn_type == RnnType.LSTM:
            outputs, (hidden, cell) = self.nn(x)       
        return hidden[-1]
        

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
                batch_first=False
            )
        else:
            raise ValueError("Unsupported RNN type")
        self.fc = nn.Linear(in_features=params.num_features, out_features=params.num_features)


    def forward(self, x, hidden, cell):
        if hidden is None or cell is None:
            output, (hidden, cell) = self.nn(x)
        else:
            output, (hidden, cell) = self.nn(x, (hidden, cell))
        prediction = self.fc(output.squeeze(0))
        return prediction, hidden, cell
    

class RNNAE(nn.Module):
    def __init__(self, device, params : Params):
        super().__init__()

        self.device = device

        self.encoder = Encoder(params=params).to(self.device)
        self.decoder = Decoder(params=params).to(self.device)

        self.seq_len = params.seq_len
        self.num_features = params.num_features


    def forward(self, x):
        h_T = self.encoder(x)
        outputs = torch.zeros(self.seq_len, 1, self.num_features).to(device)
        prediction, hidden, cell = self.decoder(h_T, None, None)
        outputs[0, :, :] = prediction.unsqueeze(0)
        for i in range(1, self.seq_len):
            prediction, hidden, cell = self.decoder(h_T, hidden, cell)
            outputs[i, :, :] = prediction.unsqueeze(0)
        return outputs
    

def train(model, dataset, batch_size, epochs, lr, device):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, _ = data

            inputs = inputs.squeeze(0).unsqueeze(1).to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass + backward pass + optimize
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    params = Params()
    params.activation_function = ActivationFunction.TANH
    params.seq_len = 30
    params.dropout = 0.1
    params.rnn_type = RnnType.LSTM
    params.num_layers = 2
    params.learning_rate = 0.001
    params.hidden_size = 4

    dir = os.path.join(os.getcwd(), 'data', 'tensors')
    files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.pt')]
    task = TaskType.Reconstruction
    dataset = TrackDataset(files=files, seq_len=params.seq_len, task=task)
    model = RNNAE(
        device=device,
        params=params
    )
    train(model, dataset, 1, 1, params.learning_rate, device)
