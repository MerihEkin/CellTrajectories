import os
import sys
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader


sys.path.append(".")

from trajectory_embeddings import trajectory_embeddings

## https://discuss.pytorch.org/t/lstm-autoencoders-in-pytorch/139727
## https://github.com/chrisvdweth/ml-toolkit/blob/master/pytorch/models/text/autoencoder/textrnnae.py


# To generate torch tensor data from dataframes
def tensor_data_generator(input_directory, output_directory):
  files = [f for f in os.listdir(input_directory) if f.endswith('.csv')]
  for file in files:
    input_file = os.path.join(input_directory, file)
    print(f'Processing {input_file}')
    df = pd.read_csv(input_file)
    df_embeddings = trajectory_embeddings(df, 1/30, 30)
    acc = np.asarray([df_embeddings.iloc[i]['a'] for i in range(len(df_embeddings))])
    vel = np.asarray([df_embeddings.iloc[i]['v'] for i in range(len(df_embeddings))])
    omega = np.asarray([df_embeddings.iloc[i]['w'] for i in range(len(df_embeddings))])
    nparray = np.concatenate((acc, vel, omega), axis=-1)
    torcharray = torch.from_numpy(nparray)
    output_file = os.path.join(output_directory, file.replace('.csv', '.pt'))
    print(f'Saved torch file to  {output_file}')
    torch.save(torcharray, output_file)


class TaskType:
  Prediction = 1
  Reconstruction = 2


class TrackDataset(Dataset):
  def __init__(self, files : list, seq_len : int, task : TaskType):
    self.data = []
    self.files = files
    self.seq_len = seq_len
    self.data_lengths = np.zeros((len(files)))

    if task == TaskType.Prediction:
      self.get_data_len = self.data_length_prediction
      self.get_item = self.get_item_prediction
    elif task == TaskType.Reconstruction:
      self.get_data_len = self.data_length_reconstruction
      self.get_item = self.get_item_reconstruction
    else:
      raise TypeError('Peanutbutter what are you doing?')
    
    self.len = 0
    self.load_data()


  def __len__(self):
    return self.len
  

  def __getitem__(self, idx):
    sequence_index = np.where(idx >= self.data_lengths)[0][-1]
    residual = int(idx - self.data_lengths[sequence_index])
    data, label = self.get_item(sequence_index, residual)
    return data.float(), label.float()


  def get_item_prediction(self, sequence_index, residual):
    data = self.data[sequence_index][residual:int(residual+self.seq_len), :]
    label = self.data[sequence_index][int(residual+self.seq_len+1), :]
    return data, label


  def get_item_reconstruction(self, sequence_index, residual):
    data = self.data[sequence_index][residual:int(residual+self.seq_len), :]
    label = self.data[sequence_index][int(residual+self.seq_len//2+1), :]
    return data, label


  def data_length_prediction(self, n):
    return n - 1 - self.seq_len


  def data_length_reconstruction(self, n):
    return n - self.seq_len


  def load_data(self):
    total_len = 0
    for i, file in enumerate(self.files):
      self.data.append(torch.load(file))
      if i == 0:
        self.data_lengths[i] = 0
      else:
        self.data_lengths[i] = total_len
      total_len +=  self.get_data_len(self.data[i].shape[0])
    self.len = int(total_len)
      

def test_data_loader():
  dir = os.path.join(os.getcwd(), 'data', 'tensors')
  files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.pt')]
  task = TaskType.Reconstruction
  seq_len = 30
  dataset = TrackDataset(files=files, seq_len=seq_len, task=task)
  dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
  for i in range(128):
    X, y = next(iter(dataloader))


def generate_tensors_from_tracks():
  input_directory = os.path.join(os.getcwd(), 'data', 'tracks')
  output_directory = os.path.join(os.getcwd(), 'data', 'tensors')
  tensor_data_generator(input_directory, output_directory)


if __name__ == '__main__':
  test_data_loader()

