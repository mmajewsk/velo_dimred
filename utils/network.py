import torch

import torch.nn as nn
import torch.nn.functional as F


class VeloEncoder(nn.Module):
    def __init__(self, input_size):
        nn.Module.__init__(self)
        self.e1 = nn.Linear(input_size, 40)
        #self.drop1 = nn.Dropout(0.1)
        self.e2 = nn.Linear(40, 10)
        #self.drop2 = nn.Dropout(0.1)
        self.e3 = nn.Linear(10, 2)
        #self.drop3 = nn.Dropout(0.1)

    def forward(self, x):
        x = self.e1(x)
        #x = self.drop1(x)
        x = F.relu(x)
        x = self.e2(x)
        #x = self.drop2(x)
        x = F.relu(x)
        x = self.e3(x)
        #x = self.drop3(x)
        return x


class VeloDecoder(nn.Module):
    def __init__(self, output_size):
        nn.Module.__init__(self)
        self.e3 = nn.Linear(2, 10)
        #self.drop3 = nn.Dropout(0.1)
        self.e2 = nn.Linear(10, 40)
        #self.drop2 = nn.Dropout(0.1)
        self.e1 = nn.Linear(40, output_size)
        #self.drop1 = nn.Dropout(0.1)

    def forward(self, x):
        x = self.e3(x)
        #x = self.drop3(x)
        x = F.relu(x)
        x = self.e2(x)
        #x = self.drop2(x)
        x = F.relu(x)
        x = self.e1(x)
        #x = self.drop1(x)
        return x

class VeloAutoencoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        torch.nn.Module.__init__(self)
        self.enc = encoder
        self.dec = decoder

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x
