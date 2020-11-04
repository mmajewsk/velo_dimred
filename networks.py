#!/usr/bin/env jupyter

#
# Splits:
# Dimentionality: [t, module, channel]
#
# Entire detector: x=t [module * channel, 1] - to little data
# Only channels: x=t*module [channel, 1]
# Only phi/R x=t*module/2 [channel, 1]

# https://gitlab.cern.ch/mmajewsk/calina.git
import sys

sys.path.append("../calina/")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from calibration_dataset import Tell1Dataset


class MyDS(Tell1Dataset):
    filename_format = "%Y-%m-%d"
    filename_regex_format = r"\d{4}-\d{2}-\d{2}.csv"


def get_dataset(path):
    data_list = MyDS.get_filepaths_from_dir(datapath)
    mds = MyDS(data_list, read=True)
    return mds


class VeloEncoderSmall(nn.Module):
    def __init__(self, input_size):
        nn.Module.__init__(self)
        self.e1 = nn.Linear(input_size, 40)
        self.e2 = nn.Linear(40, 10)
        self.e3 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.e1(x)
        x = F.relu(x)
        x = self.e2(x)
        x = F.relu(x)
        x = self.e3(x)
        x = F.relu(x)
        return x


class VeloDecoderSmall(nn.Module):
    def __init__(self, output_size):
        nn.Module.__init__(self)
        self.e3 = nn.Linear(2, 10)
        self.e2 = nn.Linear(10, 40)
        self.e1 = nn.Linear(40, output_size)

    def forward(self, x):
        x = self.e3(x)
        x = F.relu(x)
        x = self.e2(x)
        x = F.relu(x)
        x = self.e1(x)
        x = F.relu(x)
        return x


class VeloEncoderMedium(nn.Module):
    def __init__(self, input_size):
        nn.Module.__init__(self)
        self.ep1 = nn.Linear(input_size, 100)
        self.e = VeloEncoderSmall(100)

    def forward(self, x):
        x = self.ep1(x)
        x = F.relu(x)
        x = self.e(x)
        return x


class VeloDecoderMedium(nn.Module):
    def __init__(self, output_size):
        nn.Module.__init__(self)
        self.e = VeloDecoderSmall(100)
        self.ep1 = nn.Linear(100, output_size)

    def forward(self, x):
        x = self.e(x)
        x = self.ep1(x)
        x = F.relu(x)
        return x


class VeloEncoderLarge(nn.Module):
    def __init__(self, input_size):
        nn.Module.__init__(self)
        self.ep1 = nn.Linear(input_size, 1000)
        self.ep2 = nn.Linear(1000, 500)
        self.ep3 = nn.Linear(250, 100)
        self.e = VeloEncoderSmall(100)

    def forward(self, x):
        x = self.ep1(x)
        x = F.relu(x)
        x = self.ep2(x)
        x = F.relu(x)
        x = self.ep3(x)
        x = F.relu(x)
        x = self.e(x)
        return x


class VeloDecoderLarge(nn.Module):
    def __init__(self, output_size):
        nn.Module.__init__(self)
        self.e = VeloDecoderSmall(100)
        self.ep1 = nn.Linear(100, 250)
        self.ep2 = nn.Linear(250, 500)
        self.ep3 = nn.Linear(1000, output_size)

    def forward(self, x):
        x = self.e(x)
        x = self.ep1(x)
        x = F.relu(x)
        x = self.ep2(x)
        x = F.relu(x)
        x = self.ep3(x)
        x = F.relu(x)
        return x


class VeloAutoencoderLt(pl.LightningModule):
    def __init__(self, encoder, decoder):
        pl.LightningModule.__init__(self)
        self.enc = encoder
        self.dec = decoder

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        return loss
