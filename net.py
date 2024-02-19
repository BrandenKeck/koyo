# Import Libraries
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Torch Dataset Class: X = Preprocessed Text; Y = Flagged (0/1)
# Used for batch-ing data during neural network training
class KoyoDataset(Dataset):
    def __init__(self, O, D, G, Y, device='cpu'):
        self.O = torch.tensor(np.array(O)).float().to(device)
        self.D = torch.tensor(np.array(D)).float().to(device)
        self.G = torch.tensor(np.array(G)).float().to(device)
        self.Y = torch.tensor(np.array(Y)).float().to(device)
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, idx):
        O = self.O[idx]
        D = self.D[idx]
        G = self.G[idx]
        Y = self.Y[idx]
        return O, D, G, Y

# A neural network submodel
class FeedForwardNetwork(nn.Module):
    def __init__(self, dropout=0.25):
        super(FeedForwardNetwork, self).__init__()
        self.drop = nn.Dropout(p=dropout)
        self.lrelu = nn.LeakyReLU()
        self.ohid1 = nn.Linear(18, 128)
        self.ohid2 = nn.Linear(128, 64)
        self.ohid3 = nn.Linear(64, 4)
        self.dhid1 = nn.Linear(18, 128)
        self.dhid2 = nn.Linear(128, 64)
        self.dhid3 = nn.Linear(64, 4)
        self.out = nn.Linear(12, 8)
    def forward(self, o, d, g):
        o = self.lrelu(self.ohid1(o))
        o = self.drop(o)
        o = self.lrelu(self.ohid2(o))
        o = self.drop(o)
        o = self.lrelu(self.ohid3(o))
        d = self.lrelu(self.dhid1(d))
        d = self.drop(d)
        d = self.lrelu(self.dhid2(d))
        d = self.drop(d)
        d = self.lrelu(self.dhid3(d))
        c = torch.cat((o, d, g), dim=1)
        c = self.drop(c)
        return F.softmax(self.out(c)) 

# Model Class
class KoyoModel():
    def __init__(self, lr=4.12E-5, bsize=1, epochs=1000, device="cpu"):
        self.lr = lr
        self.bsize = bsize
        self.epochs = epochs
        self.device = device
        self.model = FeedForwardNetwork().to(device)
    def fit(self, O, D, G, Y, verbose=True):
        self.model.train()
        ll = nn.CrossEntropyLoss()
        oo = optim.AdamW(self.model.parameters(), lr=self.lr)
        data = KoyoDataset(O, D, G, Y, self.device)
        dataloader = DataLoader(data, batch_size=self.bsize, shuffle=True)
        for epoch in range(self.epochs):
            totalloss = 0.0
            for i, (O, D, G, Y) in enumerate(dataloader):
                Y = torch.reshape(Y, (-1, 8)).float()
                oo.zero_grad()
                y_hat = self.model(O, D, G)
                loss = ll(y_hat, Y)
                loss.backward()
                oo.step()
                totalloss += loss.item()
            totalloss = totalloss / i
            if verbose: print(f'[{epoch + 1}/{self.epochs}] Loss: {totalloss / i}')
    def predict(self, O, D, G):
        self.model.eval()
        O = torch.Tensor(O)
        D = torch.Tensor(D)
        G = torch.Tensor(G)
        y_hat = self.model(O, D, G)
        return y_hat
    def save(self, path, name):
        torch.save(self.model.state_dict(), path+name)
    def load(self, path, name):
        self.model.load_state_dict(torch.load(path+name, map_location=torch.device(self.device)))