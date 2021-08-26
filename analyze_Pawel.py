#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from datetime import datetime
datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
import neptune



#custom components
from networks import VeloDecoder, VeloEncoder, VeloAutoencoderLt
from calibration_dataset import Tell1Dataset

#trainig parameters
PARAMS = {'max_epochs': 70,
          'learning_rate': 0.02,
          'batch_size': 64,
          'gpus' : 1,
          'experiment_name' : 'small-net more-epochs standarized SGD no-dropout bigger-batches relu shuffle',
          'tags' : ['small-net', 'more-epochs', 'standarized','SGD','no-dropout','bigger-batches','relu', 'shuffle'],
          'source_files' : ['analyze_Pawel.py', 'networks.py']
         }

datasetNames = ['dfh', 'dfhr', 'dfhphi', 'dfp', 'dfpr', 'dfpphi']


class MyDS(Tell1Dataset):
	filename_format = '%Y-%m-%d'
	filename_regex_format = r'\d{4}-\d{2}-\d{2}.csv'

#loading the data
datapath = os.path.join("data", "calibrations")
data_list = MyDS.get_filepaths_from_dir(datapath)
mds = MyDS(data_list, read=True)

dfh = mds.dfh.df.iloc[:,9:]
dfh_r = mds.dfh['R'].df.iloc[:,9:]
dfh_phi = mds.dfh['phi'].df.iloc[:,9:]
dfp = mds.dfp.df.iloc[:,9:]
dfp_r = mds.dfp['R'].df.iloc[:,9:]
dfp_phi = mds.dfp['phi'].df.iloc[:,9:]


dfh_metadata = mds.dfh.df.iloc[:,:9]
dfh_r_metadata = mds.dfh['R'].df.iloc[:,:9]
dfh_phi_metadata = mds.dfh['phi'].df.iloc[:,:9]
dfp_metadata = mds.dfp.df.iloc[:,:9]
dfp_r_metadata = mds.dfp['R'].df.iloc[:,:9]
dfp_phi_metadata = mds.dfp['phi'].df.iloc[:,:9]

#scaling input data
dfh = dfh.sub(dfh.mean(1), axis=0).div(dfh.std(1), axis=0)
dfh_r = dfh_r.sub(dfh_r.mean(1), axis=0).div(dfh_r.std(1), axis=0)
dfh_phi = dfh_phi.sub(dfh_phi.mean(1), axis=0).div(dfh_phi.std(1), axis=0)
dfp = dfp.sub(dfp.mean(1), axis=0).div(dfp.std(1), axis=0)
dfp_r = dfp_r.sub(dfp_r.mean(1), axis=0).div(dfp_r.std(1), axis=0)
dfp_phi = dfp_phi.sub(dfp_phi.mean(1), axis=0).div(dfp_phi.std(1), axis=0)



def make_loader(dataset):
    train, test = train_test_split(dataset, test_size=0.2)
    train_target = torch.tensor(train.values, dtype=torch.float)
    train_data = torch.tensor(train.values, dtype=torch.float)
    test_target = torch.tensor(test.values, dtype=torch.float)
    test_data = torch.tensor(test.values, dtype=torch.float)
    train_tensor = TensorDataset(train_data, train_target)
    test_tensor = TensorDataset(test_data, test_target)
    train_loader = DataLoader(dataset = train_tensor, shuffle=True)
    test_loader = DataLoader(dataset = test_tensor, shuffle=True)
    return train_loader, test_loader


def make_model_trainer(s, neptune_logger, lr):
    s = 2048
    dec = VeloDecoder(s)
    enc = VeloEncoder(s)
    model = VeloAutoencoderLt(enc, dec, lr)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    tr = pl.Trainer(logger=neptune_logger, callbacks=[lr_monitor],  max_epochs=PARAMS['max_epochs'], gpus=PARAMS['gpus'])
    return model, tr

def run_experiment(dataset, datasetName,par):
    train_loader, test_loader = make_loader(dataset)
    s = dataset.shape[1]
    neptune_logger = NeptuneLogger(
        api_key=os.getenv('NEPTUNE_API_TOKEN'),
        project_name="pawel-drabczyk/velodimred",
        experiment_name=par['experiment_name'],
        params=par,
        tags=par['tags'] + [datasetName],
        upload_source_files= par['source_files']
    )
    model, tr = make_model_trainer(s, neptune_logger, par['learning_rate'])
    tr.fit(model, train_loader, test_loader)

    torch.save(model, os.path.join('models', PARAMS['experiment_name'], datasetName,"trained_model.ckpt" ) )
    neptune_logger.experiment.log_artifact(os.path.join('models', PARAMS['experiment_name'], datasetName,"trained_model.ckpt" ))

if __name__ == "__main__":
	for d in datasetNames:
	    if not os.path.exists(os.path.join('models', PARAMS['experiment_name'], d)):
	        os.makedirs(os.path.join('models', PARAMS['experiment_name'], d))

	run_experiment(dfh, 'dfh', PARAMS)
	run_experiment(dfh_r, 'dfhr', PARAMS)
	run_experiment(dfh_phi, 'dfhphi', PARAMS)
	run_experiment(dfp, 'dfp', PARAMS)
	run_experiment(dfp_r, 'dfpr', PARAMS)
	run_experiment(dfp_phi, 'dfpphi', PARAMS)
