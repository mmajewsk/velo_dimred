#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import torch
import logging
import os
import neptune
from calibration_dataset import Tell1Dataset
logging.basicConfig(level=logging.INFO)

PARAMS = {'max_epochs': 1,
          'learning_rate': 0.05,
          'batch_size': 64,
          'gpus': 1,
          'experiment_name': 'small-net more-epochs standarized SGD no-dropout bigger-batches relu shuffle',
          'tags': ['small-net', 'more-epochs', 'standarized', 'SGD', 'no-dropout', 'bigger-batches', 'relu', 'shuffle'],
          'source_files': ['analyze_Pawel.py', 'networks.py'],
          'experiment_id': 'VEL-374'
}

datasetNames = ['dfh', 'dfhr', 'dfhphi', 'dfp', 'dfpr', 'dfpphi']


class MyDS(Tell1Dataset):
    filename_format = '%Y-%m-%d'
    filename_regex_format = r'\d{4}-\d{2}-\d{2}.csv'


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

#scaling input data
dfh = dfh.sub(dfh.mean(1), axis=0).div(dfh.std(1), axis=0)
dfh_r = dfh_r.sub(dfh_r.mean(1), axis=0).div(dfh_r.std(1), axis=0)
dfh_phi = dfh_phi.sub(dfh_phi.mean(1), axis=0).div(dfh_phi.std(1), axis=0)
dfp = dfp.sub(dfp.mean(1), axis=0).div(dfp.std(1), axis=0)
dfp_r = dfp_r.sub(dfp_r.mean(1), axis=0).div(dfp_r.std(1), axis=0)
dfp_phi = dfp_phi.sub(dfp_phi.mean(1), axis=0).div(dfp_phi.std(1), axis=0)

dfh_metadata = mds.dfh.df.iloc[:, :9]
dfh_r_metadata = mds.dfh['R'].df.iloc[:, :9]
dfh_phi_metadata = mds.dfh['phi'].df.iloc[:, :9]
dfp_metadata = mds.dfp.df.iloc[:, :9]
dfp_r_metadata = mds.dfp['R'].df.iloc[:, :9]
dfp_phi_metadata = mds.dfp['phi'].df.iloc[:, :9]


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


def plot(dataset, datasetName, metadata, exp_name, exp_id):
    model_path = os.path.join('models', exp_name, datasetName, 'trained_model.ckpt')

    if not os.path.exists(model_path):
        logging.info('{} does not exists, exiting'.format(model_path) )
        exit()

    model = torch.load(model_path)

    reducedData = model.enc.forward(torch.tensor(dataset.values, dtype=torch.float))
    reducedData = reducedData.detach().numpy()

    x2DList = []
    y2DList = []
    sensorNumberList = [0]
   
    tempSensor = 0
    counter = 0
    tempX = []
    tempY = []    
    for sensor in metadata['sensor']:
        if int(sensor) == tempSensor:
            tempX.append(reducedData[counter][0])
            tempY.append(reducedData[counter][1])
        else:
            x2DList.append(tempX)
            y2DList.append(tempY)
            tempX = [reducedData[counter][0]]
            tempY = [reducedData[counter][1]]
            sensorNumberList.append(int(sensor))
        counter = counter + 1
        tempSensor = sensor
    x2DList.append(tempX)
    y2DList.append(tempY)

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(111)
    alpha = 0.4
    cmap = get_cmap(len(sensorNumberList))
    for sensorNumber in range(len(sensorNumberList)):
        plt.scatter(x2DList[sensorNumber], y2DList[sensorNumber], c=cmap(sensorNumber), edgecolor='none', alpha=alpha,
                    label=sensorNumberList[sensorNumber])

    plt.xlabel('Reduced variable 1')
    plt.ylabel('Reduced variable 2')
    plt.legend(title="Module nr.", ncol=5)
    plt.show()
    fig.savefig(os.path.join('models', exp_name, datasetName, 'reduced.png'))
    
    project = neptune.init('pawel-drabczyk/velodimred')
    my_exp = project.get_experiments(id=exp_id)[0]
    my_exp.append_tag('plot-added')
    my_exp.log_image('reducedData', fig, image_name='reducedData')

#plot(dfh, 'dfh', dfh_metadata, PARAMS['experiment_name'], PARAMS['experiment_id'])
plot(dfh_r, 'dfhr', dfh_r_metadata, PARAMS['experiment_name'], 'VEL-371')
# plot(dfh_phi, 'dfhphi', dfh_phi_metadata, PARAMS['experiment_name'], 'VEL-372')
# plot(dfp, 'dfp', dfp_metadata, PARAMS['experiment_name'], 'VEL-373')
# plot(dfp_r, 'dfpr', dfp_r_metadata, PARAMS['experiment_name'], 'VEL-374')
# plot(dfp_phi, 'dfpphi', dfp_phi_metadata, PARAMS['experiment_name'], 'VEL-375')

#plot(dfp_r, 'dfpr', dfp_r_metadata, PARAMS['experiment_name'], PARAMS['experiment_id'])
#plot(dfp_phi, 'dfpphi', dfp_phi_metadata, PARAMS['experiment_name'], PARAMS['experiment_id'])


#run_experiment(dfh, 'dfh', dfh_sensor_numbers, PARAMS)
#run_experiment(dfh_r, 'dfhr', PARAMS)
#run_experiment(dfh_phi, 'dfhphi', PARAMS)
#run_experiment(dfp, 'dfp', PARAMS)
#run_experiment(dfp_r, 'dfpr', PARAMS)
#run_experiment(dfp_phi, 'dfpphi', PARAMS)
