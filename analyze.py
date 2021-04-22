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

# <h1> VELO detector - solving dimensional reduction problem using autoencoders and PCA</h1>
# <h2>Learn how PyTorch and scikit-learn help in preparing callibration data for CERN's analyses</h2>
# <h3> Authors: Tymoteusz Ciesielski, Paweł Drabczyk, Aleksander Morgała</h3>
# <h4> AGH UST 2020/2021 Faculty of Physics and Applied Computer Science</h4>
# <h4> Project realised for the Python in the Enterprise course. </h4>
#

# <h3>1. Introduction</h3>
#
# What do you see when you think about particle collisions in CERN? Most people imagine a lot of colorful track and clusters in the detector like this:
#
# <img src='https://physicsworld.com/wp-content/uploads/2018/08/LHCb-collision-635x372.png' alt='Sophia Chen, Charmed baryon puzzles particle physicists by living longer' width="1000">
#
# In reallity there is a long way from collision to this visualisation. Lets start with a short introduction on the LHCb detector. The LHCb detector is divided into parts: vertex locator, electromagnetic calorimeter, hadronic calorimeter and muon system. Each part provides different kind of information about partcile (or lack of information, what is also an information). In this analysis we focus on VELO (VErtex LOcator), the most precise tracking system in the world. 
#
# <img src='http://cds.cern.ch/record/1017398/files/velo-2007-003_01.jpg?version=1' alt='Paula Collins, Velo constaint system installation(site C)' width="1000">
#
# All kind of information are gathered as an electric signal and whereever there is electric signal there is a...electric noise. You can minimalise the noise, but you can never get rid of it in 100%. So, after you come to terms with it you have to measure the noise in order to distinguish the noise from the real signal from particles. This is done during callibration measurement, when the detector is turned on, but there are no collisions happening. The measured noise looks like this:
#

# ![image.png](attachment:image.png)

# Before we analyse above picture it is good to know that electric noise amplitude have a gaussian distribution. Each gaussian distribution have certain average value. Interpretation of this value at the above picture is the following: each sensor have certain level of average electric voltage which is present all the time. This value is called 'pedestal'. Therefore, it is enough to just subtract pedestals from measured voltage value. 

# ![image.png](attachment:image.png)

# After the sutraction of the pedestals the average value of noise for all of the sensors is equal to 0. Now we can measure the standard deviation of the noise distribution for each sensor. However, the distribution is not symetrical so we define different standard deviation for positive and negative values of voltage. 
#
# How do we use the characteristics of noise distribution? Physicists pay a lot of attention to avoid false positive (nobody want to withdraw false discovery in shame). Therefore, the five-sigma or even six-sigma significance rule is used. It means that only voltages distant from pedestal value for five or six standard deviations are considered as not noise effects (real particles). In this case five-sigma significance means that the chance that noise can be indentified as a real particle is like one-in-a-milion (one-in-a half-bilion for six sigma).
#
# The summary of the characteristics of the electric noise (with working names):
# <ol>
#     <li> dfp - 'pedestals', the average values noise distribution </li>
#     <li> dfh - high hit threshold, the standard deviation of noise distribution for voltages above pedestal </li>
#     <li> dfl - low hit threshold, the standard deviation of noise distribution for voltages under pedestal </li>
# </ol>
#
# Ok, so we have the electric noise characterised by callibration data. Now physicists can run many different analyses looking for any kind of correlation between noise and final signal measured. For example they can use neural networks, but... the number of dimensions in callibration data is far too big. We have 4096 sensors present in VELO detector, and callibration data were taken several times. If we multiply those numbers we receive something even more inadequate to be input for neural network.
#
# Therefore we performed dimension reduction for callibration data. The output of the reduction can be used by other analyses. We have applied two approaches:
# <ol>
#     <li> Principal Component Analysis (PCA) </li>
#     <li> Autoencoders </li>
# </ol>
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
# Requirement.txt can be found in the repository. In order install all necessary packages you should run:
#
# pip install -r requirements.txt
#
# Adding Calina library to pythonpath is also necessary. If you are using using anaconda you can just type:
#
# conda develop Calina_path

# <h3> 4. Code example </h3>

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
import neptune
import zipfile

# networks.py contains our custom architectures for autoencoders. They are built upon the pl.LightningModule

from networks import VeloDecoder, VeloEncoder, VeloAutoencoderLt
from calibration_dataset import Tell1Dataset

# Below are all the necessary parameters for the training process, creating plots and saving them in Neptune.

#trainig parameters
PARAMS = {'max_epochs': 50,
          'learning_rate': 0.02,
          'batch_size': 64,
          'gpus': 1,
          'experiment_name': 'small-net more-epochs standarized2 SGD no-dropout bigger-batches relu shuffle',
          'tags': ['small-net', 'normal-epochs', 'standarized2', 'SGD', 'no-dropout', 'bigger-batches', 'relu', 'shuffle'],
          'source_files': ['analyze.pynb', 'networks.py'],
          'experiment_id': 'VEL-370'
}


# Some help for extracting the data.

class MyDS(Tell1Dataset):
    filename_format = '%Y-%m-%d'
    filename_regex_format = r'\d{4}-\d{2}-\d{2}.csv'


# Loading the data from all sources and channels. We also scale the input data.

# +
#loading the data
datapath = os.path.join("data", "calibrations")
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


#scaling input data
# dfh = dfh.sub(dfh.mean(1), axis=0).div(dfh.std(1), axis=0)
# dfh_r = dfh_r.sub(dfh_r.mean(1), axis=0).div(dfh_r.std(1), axis=0)
# dfh_phi = dfh_phi.sub(dfh_phi.mean(1), axis=0).div(dfh_phi.std(1), axis=0)
# dfp = dfp.sub(dfp.mean(1), axis=0).div(dfp.std(1), axis=0)
# dfp_r = dfp_r.sub(dfp_r.mean(1), axis=0).div(dfp_r.std(1), axis=0)
# dfp_phi = dfp_phi.sub(dfp_phi.mean(1), axis=0).div(dfp_phi.std(1), axis=0)


# print('type(dfh_scaled)')
# print(type(dfh_scaled))
# print('type(dfh)')
# print(type(dfh))

# print('mean standard scaler')
# print(dfh_scaled.mean(axis=0))
# print('mean own method')
# print(dfh.mean(axis=0))
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

run_experiment(dfh, 'dfh', PARAMS, dfh_metadata)
run_experiment(dfh_r, 'dfhr', PARAMS, dfh_r_metadata)
run_experiment(dfh_phi, 'dfhphi', PARAMS, dfh_phi_metadata)
run_experiment(dfp, 'dfp', PARAMS, dfp_metadata)
run_experiment(dfp_r, 'dfpr', PARAMS, dfp_r_metadata)
run_experiment(dfp_phi, 'dfpphi', PARAMS, dfp_phi_metadata)
# -

# The cell below opens existing experiment, adding the slider plots and cluster plots.

# +
#reopen_experiment(dfh, 'dfh', PARAMS, dfh_metadata)
#reopen_experiment(dfh_r, 'dfhr', PARAMS, dfh_r_metadata)
#reopen_experiment(dfh_phi, 'dfhphi', PARAMS, dfh_phi_metadata)
#reopen_experiment(dfp, 'dfp', PARAMS, dfp_metadata)
#reopen_experiment(dfp_r, 'dfpr', PARAMS, dfp_r_metadata)
#reopen_experiment(dfp_phi, 'dfpphi', PARAMS, dfp_phi_metadata)
# -

# The results of dimension reduction for upper hit thre

