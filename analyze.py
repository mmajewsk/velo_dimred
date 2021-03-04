# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# <h1> VELO detector - dimensional reduction problem using autoencoders and PCA</h1>
# <h3> Authors: Tymoteusz Ciesielski, Paweł Drabczyk, Aleksander Morgała </h3>
# <h4> AGH UST 2020/2021 Faculty of Physics and Applied Computer Science</h4>
# <h4> Project realised for the Python in the Enterprise course. </h4>
#

# <h3>1. Introduction</h3>
# Our task was to reduce the number of dimensions in the VELO detector. Each dimension of the problem represents one sensor from 4096 present. We have applied two approaches:
# <ol>
#     <li> Principal Component Analysis (PCA) </li>
#     <li> Autoencoders </li>
# </ol>
#
# <a href=https://lhcb-public.web.cern.ch/en/detector/VELO-en.html>Link to more information about VELO.</a>
#
# In the code below, you can find the application of the autoencoders.
#

# <h3> 2. Data </h3>
# We were using the data from here:
#
# [insert some links etc]
#
# If you want to run the code, you should place the folder with the data in your working directory.

# <h3> 3. Technologies </h3>
# <ul>
#     <li> <b>Python</b> (PyTorch) - provides the basic architecture for the neural networks we were using</li>
#     <li> <b>Neptune AI </b> - to keep track of the training process, register networks architectures and save the parameters</li>
#     <li> <b>Calina</b> library</li>
# </ul>
#     

# <h3> 4. Code example </h3>

import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from datetime import datetime
datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
import plotly.express as px
import pandas as pd
import neptune
import zipfile

# networks.py contains our custom architectures for autoencoders. They are built upon the pl.LightningModule

from networks import VeloDecoder, VeloEncoder, VeloAutoencoderLt
from calibration_dataset import Tell1Dataset

# Below are all the necessary parameters for the training process, creating plots and saving them in Neptune.

#trainig parameters
PARAMS = {'max_epochs': 1,
          'learning_rate': 0.02,
          'batch_size': 64,
          'gpus': 1,
          'experiment_name': 'small-net more-epochs standarized SGD no-dropout bigger-batches relu shuffle',
          'tags': ['small-net', 'normal-epochs', 'standarized', 'SGD', 'no-dropout', 'bigger-batches', 'relu', 'shuffle'],
          'source_files': ['small-net', 'more-epochs', 'standarized', 'SGD', 'no-dropout', 'bigger-batches', 'relu', 'shuffle'],
          'experiment_id': 'VEL-398'
}


# Some help for extracting the data.

class MyDS(Tell1Dataset):
    filename_format = '%Y-%m-%d'
    filename_regex_format = r'\d{4}-\d{2}-\d{2}.csv'


# Loading the data from all soutces and channels. We also scale the input data.

# +
#loading the data
datapath = os.path.join("data", "calibrations")
data_list = MyDS.get_filepaths_from_dir(datapath)
mds = MyDS(data_list, read=True)

dfh = mds.dfh.df.iloc[:, 9:]
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


#scaling input data
dfh = dfh.sub(dfh.mean(1), axis=0).div(dfh.std(1), axis=0)
dfh_r = dfh_r.sub(dfh_r.mean(1), axis=0).div(dfh_r.std(1), axis=0)
dfh_phi = dfh_phi.sub(dfh_phi.mean(1), axis=0).div(dfh_phi.std(1), axis=0)
dfp = dfp.sub(dfp.mean(1), axis=0).div(dfp.std(1), axis=0)
dfp_r = dfp_r.sub(dfp_r.mean(1), axis=0).div(dfp_r.std(1), axis=0)
dfp_phi = dfp_phi.sub(dfp_phi.mean(1), axis=0).div(dfp_phi.std(1), axis=0)

# -

# Create loaders for the training.

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


# Create trainers for the network, with some set training parameters.

def make_model_trainer(s, neptune_logger, lr):
    #s = 2048
    dec = VeloDecoder(s)
    enc = VeloEncoder(s)
    model = VeloAutoencoderLt(enc, dec, lr)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    tr = pl.Trainer(logger=neptune_logger, callbacks=[lr_monitor],  max_epochs=PARAMS['max_epochs'],
                    gpus=PARAMS['gpus'])
    return model, tr


# Function creating the slider plot to see the evolution of autoencoder results

def slider_plot(dataset, datasetName, metadata, model):
    reducedData = model.enc.forward(torch.tensor(dataset.values, dtype=torch.float))
    reducedData = reducedData.detach().numpy()
     
    indexesList = metadata.index.values.tolist()
    xyDF = pd.DataFrame(reducedData, index=indexesList, columns=['x', 'y'])
    resultDF = pd.concat([metadata, xyDF], axis=1)
    resultDF["datetime"] = resultDF["datetime"].astype(str)

    fig = px.scatter(resultDF, x="x", y="y", animation_frame="datetime", animation_group="sensor", color="sensor")
    fig["layout"].pop("updatemenus") # optional, drop animation buttons
    fig.show()
    return fig


# Function creating the plot being the result of the training. All the channels can be later turned on/off in a custom way

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


# Function connects to Neptune, starts a new experiment there and starts running the training process for the network. It then creates all the plots and saves them in the Neptune as well.

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


# The function was used to reopen experiments, and add the cluster plots and slider plots, which was added at the end of the project.

def reopen_experiment(dataset, datasetName, par, metadata):
    exp_id = par['experiment_id']
    exp_name = par['experiment_name']
    model_path = os.path.join('models', exp_name, datasetName)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    project = neptune.init('pawel-drabczyk/velodimred')
    my_exp = project.get_experiments(id=exp_id)[0]
    my_exp.download_artifact('trained_model.ckpt', model_path)
    my_exp.download_sources('networks.py')
    with zipfile.ZipFile("networks.py.zip","r") as zip_ref:
        zip_ref.extractall()
    from networks import VeloDecoder, VeloEncoder, VeloAutoencoderLt
        
    model = torch.load(os.path.join(model_path,'trained_model.ckpt'))
    
    fig = slider_plot(dataset, datasetName, metadata, model)
    fig.write_html(os.path.join(model_path, 'slider_plot.html'))
    fig.write_image(os.path.join(model_path, 'slider_plot.png'))   
    my_exp.log_image('slider_plot',os.path.join(model_path, 'slider_plot.png'))    
    fig = clustering_plot(dataset, datasetName, metadata, model)
    fig.write_html(os.path.join(model_path, 'clustering_plot.html'))
    fig.write_image(os.path.join(model_path, 'clustering_plot.png'))    
    my_exp.log_image('clustering_plot', os.path.join(model_path, 'clustering_plot.png'))        
    
    my_exp.log_artifact(os.path.join(model_path, "slider_plot.html"))
    my_exp.log_artifact(os.path.join(model_path, "clustering_plot.html"))
    my_exp.append_tag('interactive')


# The cell below runs te training process for all the datasets, for the current experiment, for the current network configuration.

# +
datasetNames = ['dfh', 'dfhr', 'dfhphi', 'dfp', 'dfpr', 'dfpphi']

#run_experiment(dfh, 'dfh', PARAMS, dfh_metadata)
#run_experiment(dfh_r, 'dfhr', PARAMS, dfh_r_metadata)
#run_experiment(dfh_phi, 'dfhphi', PARAMS, dfh_phi_metadata)
#run_experiment(dfp, 'dfp', PARAMS, dfp_metadata)
#run_experiment(dfp_r, 'dfpr', PARAMS, dfp_r_metadata)
#run_experiment(dfp_phi, 'dfpphi', PARAMS, dfp_phi_metadata)
# -

# The cell below opens existing experiment, adding the slider plots and cluster plots.

#reopen_experiment(dfh, 'dfh', PARAMS, dfh_metadata)
#reopen_experiment(dfh_r, 'dfhr', PARAMS, dfh_r_metadata)
#reopen_experiment(dfh_phi, 'dfhphi', PARAMS, dfh_phi_metadata)
#reopen_experiment(dfp, 'dfp', PARAMS, dfp_metadata)
#reopen_experiment(dfp_r, 'dfpr', PARAMS, dfp_r_metadata)
reopen_experiment(dfp_phi, 'dfpphi', PARAMS, dfp_phi_metadata)
