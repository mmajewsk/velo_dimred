# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# <h1> Autoencoders</h1>

# #### Installation
#     
# requirement.txt can be found in the repository. In order install all necessary packages you should run:
#
# pip install -r requirements.txt
#
# Adding Calina library to pythonpath is also necessary. If you are using using anaconda you can just type:
#
# conda develop Calina_path

# #### Importing required modules

import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from datetime import datetime
datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
import plotly.express as px
import pandas as pd
import neptune.new as neptune
import zipfile

# #### networks.py contains our custom architectures for autoencoders. They are built upon the pl.LightningModule

from networks import VeloDecoder, VeloEncoder, VeloAutoencoderLt
from calina_dataset.calibration_dataset import Tell1Dataset

# #### Below are all the necessary parameters for the training process, creating plots and saving them in Neptune.

#trainig parameters
PARAMS = {'max_epochs': 50,
          'learning_rate': 0.02,
          'batch_size': 64,
          'gpus': 1,
          'experiment_name': 'small-net more-epochs standarized SGD no-dropout bigger-batches relu shuffle',
          'tags': ['small-net', 'more-epochs', 'standarized', 'SGD', 'no-dropout', 'bigger-batches', 'relu', 'shuffle'],
          'source_files': ['analyze.ipynb', 'networks.py'],
          'experiment_id': 'VEL-371'
}


# #### Class for extracting the data.

class MyDS(Tell1Dataset):
    filename_format = '%Y-%m-%d'
    filename_regex_format = r'\d{4}-\d{2}-\d{2}.csv'


# #### Loading the data from all sources and channels. We also standardize the input data.

# +
datapath = os.path.join("../../data", "calibrations")
data_list = MyDS.get_filepaths_from_dir(datapath)
mds = MyDS(data_list, read=True)

dfh = mds.dfh.df.iloc[:, 9:]
print(dfh)
dfh_r = mds.dfh['R'].df.iloc[:, 9:]
dfh_phi = mds.dfh['phi'].df.iloc[:, 9:]
dfp = mds.dfp.df.iloc[:, 9:]
dfp_r = mds.dfp['R'].df.iloc[:, 9:]
dfp_phi = mds.dfp['phi'].df.iloc[:, 9:]

dfh_metadata = mds.dfh.df.iloc[:, :9]
dfh_r_metadata = mds.dfh['R'].df.iloc[:, :9]
dfh_phi_metadata = mds.dfh['phi'].df.iloc[:, :9]
dfp_metadata = mds.dfp.df.iloc[:, :9]
dfp_r_metadata = mds.dfp['R'].df.iloc[:, :9]
dfp_phi_metadata = mds.dfp['phi'].df.iloc[:, :9]

scaler = preprocessing.StandardScaler()
dfh_scaled = scaler.fit_transform(dfh)
dfh_r_scaled = scaler.fit_transform(dfh_r)
dfh_phi_scaled = scaler.fit_transform(dfh_phi)
dfp_scaled = scaler.fit_transform(dfp)
dfp_r_scaled = scaler.fit_transform(dfp_r)
dfp_phi_scaled = scaler.fit_transform(dfp_phi)

dfh = pd.DataFrame(dfh_scaled, index=dfh.index, columns=dfh.columns)
dfh_r = pd.DataFrame(dfh_r_scaled, index=dfh_r.index, columns=dfh_r.columns)
dfh_phi = pd.DataFrame(dfh_phi_scaled, index=dfh_phi.index, columns=dfh_phi.columns)
dfp = pd.DataFrame(dfp_scaled, index=dfp.index, columns=dfp.columns)
dfp_r = pd.DataFrame(dfp_r_scaled, index=dfp_r.index, columns=dfp_r.columns)
dfp_phi = pd.DataFrame(dfp_phi_scaled, index=dfp_phi.index, columns=dfp_phi.columns)


# -

# #### Creating loaders for the training.

def make_loader(dataset):
    train, test = train_test_split(dataset, test_size=0.2)
    train_target = torch.tensor(train.values, dtype=torch.float)
    train_data = torch.tensor(train.values, dtype=torch.float)
    test_target = torch.tensor(test.values, dtype=torch.float)
    test_data = torch.tensor(test.values, dtype=torch.float)
    train_tensor = TensorDataset(train_data, train_target)
    test_tensor = TensorDataset(test_data, test_target)
    train_loader = DataLoader(dataset=train_tensor, shuffle=True)
    test_loader = DataLoader(dataset=test_tensor, shuffle=True)
    return train_loader, test_loader


# #### Creating trainers for the network, with some set training parameters.

def make_model_trainer(s, neptune_logger, lr):
    #s = 2048
    dec = VeloDecoder(s)
    enc = VeloEncoder(s)
    model = VeloAutoencoderLt(enc, dec, lr)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    tr = pl.Trainer(logger=neptune_logger, callbacks=[lr_monitor],  max_epochs=PARAMS['max_epochs'],
                    gpus=PARAMS['gpus'])
    return model, tr


# #### Function creating the slider plot to see the evolution of autoencoder results

def slider_plot(dataset, datasetName, metadata, model):
    reducedData = model.enc.forward(torch.tensor(dataset.values, dtype=torch.float))
    reducedData = reducedData.detach().numpy()
     
    indexesList = metadata.index.values.tolist()
    xyDF = pd.DataFrame(reducedData, index=indexesList, columns=['x', 'y'])
    resultDF = pd.concat([metadata, xyDF], axis=1)
    resultDF["datetime"] = resultDF["datetime"].astype(str)

    fig = px.scatter(resultDF, x="x", y="y", animation_frame="datetime", animation_group="sensor", color="sensor")
    fig["layout"].pop("updatemenus") # optional, drop animation buttons
    fig.update_xaxes(range=[1.15*resultDF['x'].min(), 1.15*resultDF['y'].max()])
    fig.update_yaxes(range=[1.15*resultDF['x'].min(), 1.15*resultDF['y'].max()])
    fig.show()
    return fig


# #### Function creating the plot being the result of the training. All the channels can be later turned on/off in a custom way

def clustering_plot(dataset, datasetName, metadata, model):
    
    reducedData = model.enc.forward(torch.tensor(dataset.values, dtype=torch.float))
    reducedData = reducedData.detach().numpy()
     
    indexesList = metadata.index.values.tolist()
    xyDF = pd.DataFrame(reducedData, index=indexesList, columns=['x', 'y'])
    resultDF = pd.concat([metadata, xyDF], axis=1)
    resultDF["datetime"] = resultDF["datetime"].astype(str)
    resultDF["sensor"] = resultDF["sensor"].astype(str)
    print(resultDF)
    
    fig = px.scatter(resultDF, x="x", y="y", color='sensor', opacity=0.5)
    fig.show(renderer="notebook")
    return fig


# #### Function connects to Neptune, starts a new experiment there and starts running the training process for the network. It then creates all the plots and saves them in the Neptune as well.

def run_experiment(dataset, datasetName, par, metadata):
    model_path = os.path.join('models', par['experiment_name'], datasetName)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    train_loader, test_loader = make_loader(dataset)
    s = dataset.shape[1]
    neptune_logger = NeptuneLogger(
        api_key=os.getenv('NEPTUNE_API_TOKEN'),
        project_name="pawel-drabczyk/velodimred",
        experiment_name=par['experiment_name'],
        params=par,
        tags=par['tags'] + [datasetName] + ['interactive'],
        upload_source_files=par['source_files']
    )
    model, tr = make_model_trainer(s, neptune_logger, par['learning_rate'])
    tr.fit(model, train_loader, test_loader)

    torch.save(model, os.path.join('models', PARAMS['experiment_name'], datasetName, "trained_model.ckpt"))
    neptune_logger.experiment.log_artifact(os.path.join(model_path, "trained_model.ckpt"))
    
    fig = slider_plot(dataset, datasetName, metadata, model)
    fig.write_html(os.path.join(model_path, 'slider_plot.html'))
    fig.write_image(os.path.join(model_path, 'slider_plot.png'))   
    neptune_logger.experiment.log_image('slider_plot',os.path.join(model_path, 'slider_plot.png'))    
    fig = clustering_plot(dataset, datasetName, metadata, model)
    fig.write_html(os.path.join(model_path, 'clustering_plot.html'))
    fig.write_image(os.path.join(model_path, 'clustering_plot.png'))    
    neptune_logger.experiment.log_image('clustering_plot', os.path.join(model_path, 'clustering_plot.png'))        
    
    neptune_logger.experiment.log_artifact(os.path.join(model_path, "slider_plot.html"))
    neptune_logger.experiment.log_artifact(os.path.join(model_path, "clustering_plot.html"))


# #### The function was used to reopen experiments, and add the cluster plots and slider plots, which was added at the end of the project.

def reopen_experiment(dataset, datasetName, par, metadata):
    exp_id = par['experiment_id']
    exp_name = par['experiment_name']
    model_path = os.path.join('models', exp_name, datasetName)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    run = neptune.init('pawel-drabczyk/velodimred', run=PARAMS['experiment_id'], mode="read-only", capture_stderr=False, capture_stdout=False)
    run['artifacts/trained_model.ckpt'].download()
    with zipfile.ZipFile("files.zip","r") as zip_ref:
        zip_ref.extractall('legacy_network')
        
    # uncomment the following line when the dependancy injection problem is solved
    # from legacy_network.networks import VeloDecoder, VeloEncoder, VeloAutoencoderLt
    from networks import VeloDecoder, VeloEncoder, VeloAutoencoderLt

    model = torch.load(os.path.join(model_path,'trained_model.ckpt'), map_location=torch.device('cpu'))
    
    fig = slider_plot(dataset, datasetName, metadata, model)
    fig.write_html(os.path.join(model_path, 'slider_plot.html'))
    fig.write_image(os.path.join(model_path, 'slider_plot.png'))   
  
    fig = clustering_plot(dataset, datasetName, metadata, model)
    fig.write_html(os.path.join(model_path, 'clustering_plot.html'))
    fig.write_image(os.path.join(model_path, 'clustering_plot.png'))    



# #### The cell below runs te training process for all the datasets, for the current experiment, for the current network configuration.

# +
datasetNames = ['dfh', 'dfhr', 'dfhphi', 'dfp', 'dfpr', 'dfpphi']

#run_experiment(dfh, 'dfh', PARAMS, dfh_metadata)
#run_experiment(dfh_r, 'dfhr', PARAMS, dfh_r_metadata)
#run_experiment(dfh_phi, 'dfhphi', PARAMS, dfh_phi_metadata)
#run_experiment(dfp, 'dfp', PARAMS, dfp_metadata)
#run_experiment(dfp_r, 'dfpr', PARAMS, dfp_r_metadata)
#run_experiment(dfp_phi, 'dfpphi', PARAMS, dfp_phi_metadata)
# -

# #### The cell below opens existing experiment, adding the slider plots and cluster plots.

#reopen_experiment(dfh, 'dfh', PARAMS, dfh_metadata)
reopen_experiment(dfh_r, 'dfhr', PARAMS, dfh_r_metadata)
#reopen_experiment(dfh_phi, 'dfhphi', PARAMS, dfh_phi_metadata)
#reopen_experiment(dfp, 'dfp', PARAMS, dfp_metadata)
#reopen_experiment(dfp_r, 'dfpr', PARAMS, dfp_r_metadata)
#reopen_experiment(dfp_phi, 'dfpphi', PARAMS, dfp_phi_metadata)
