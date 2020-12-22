# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ### Importing required modules

import pandas as pd
# %load_ext autoreload
# %autoreload 2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ### Loading data into dataset

# +

from calibration_dataset import Tell1Dataset

class MyDS(Tell1Dataset):
    filename_format = '%Y-%m-%d'
    filename_regex_format = r'\d{4}-\d{2}-\d{2}.csv'

datapath = "../../data/calibrations/"
data_list = MyDS.get_filepaths_from_dir(datapath)
mds = MyDS(data_list, read=True)



# -

# ### Seperating data

data = {'hit threshold':mds.dfh.df,'pedestal' : mds.dfp.df,'low threshold': mds.dfl.df}

# ### Clearing data

for key in data:
    print(key)
    data[key] = data[key].drop(['Zmod','slot_label','mod_nr','sensor_number','type','datetime'],axis=1)
    print(data[key].sensor_type.unique())
    data[key] = {'phi':data[key][data[key]['sensor_type'] == 'phi'],\
                    'r_phi':data[key][data[key]['sensor_type'] == 'R']}
    for typ in data[key]:
        data[key][typ] = data[key][typ].drop(['mod_type','sensor_type'],axis=1)

# ### Importing modules needed for PCA

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random


# ### Sets color map for plot, can be changed by editing 
# ##### cm = plt.get_cmap(new color map)
#

def set_color(plot):
    num_colors = 30
    cm = plt.get_cmap('jet')
    cNorm  = mpl.colors.Normalize(vmin=0, vmax=num_colors-1)
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cm)
    colors = [scalarMap.to_rgba(i) for i in range(num_colors)]
    random.shuffle(colors,lambda: 0.1)
    plot.set_prop_cycle("color",colors)


# ### Data normalization and PCA

def full_pca(data,percent):
    data = StandardScaler().fit_transform(data)
    data_transponsed = data.transpose()
    pca = PCA(n_components = percent,svd_solver = 'auto')
    result = pca.fit_transform(data_transponsed)
    column_names = [f"Principal component {x+1}" for x in range(len(result[0]))]
    row_names = [f"Channel {x}" for x in range(len(result))]
    return pd.DataFrame(result,columns = column_names,index = row_names)


# ### Setting plot size and number of primal components 

plt.rcParams['figure.figsize'] = [16, 9*15]
procentage_or_num_of_comp = 2


# ### Gets single sensor data, making PCA and scattering it at plot
# #### alpha - sets transparency of the points

def scatter_data(single_data): 
    alpha = 0.4
    for sensor_data_key in single_data:
        dataset = single_data[sensor_data_key]
        dataset_after_pca = full_pca(dataset,procentage_or_num_of_comp)
        scatter = plt.scatter(dataset_after_pca.iloc[:,0], dataset_after_pca.iloc[:,1], edgecolor='none', alpha=alpha,label=sensor_data_key)
        plt.legend(title="Module nr.")


# ### Separating module type data into single module

def draw_a_plot(sensor, mod_key):
    for sensor_key in sensor:
        single_data = sensor[sensor_key]
        plot = plt.subplot(draw_a_plot.position, title=f'{mod_key} - {sensor_key}')
        set_color(plot)
        single_data = {k: v.drop('sensor',axis=1) for k, v in single_data.groupby('sensor')}
        scatter_data(single_data)
        plt.xlabel('Principal component 1')
        plt.ylabel('Principal component 2')
        draw_a_plot.position+=1


# ### Separates data into modules types data then plotting it

def do_a_pca_and_draw_a_plot(data):
    draw_a_plot.position = 911
    plt.suptitle("PCA results",fontsize=16)
    for mod_key in data:
        draw_a_plot(data[mod_key],mod_key)
        plt.tight_layout()



import plotly.express as px

d = data['hit threshold']['phi']

# +
single_data = {k: v.drop('sensor',axis=1) for k, v in d.groupby('sensor')}
alpha = 0.4
thisrun = []
for sensor_data_key in single_data:
    dataset = single_data[sensor_data_key]
    
    dataset_after_pca = full_pca(dataset,procentage_or_num_of_comp)
    dataset_after_pca["sensor"] = str(int(sensor_data_key))
    #scatter = plt.scatter(dataset_after_pca.iloc[:,0], dataset_after_pca.iloc[:,1], edgecolor='none', alpha=alpha,label=sensor_data_key)
    #plt.legend(title="Module nr.")
    thisrun.append(dataset_after_pca)


# -

alldat = pd.concat(thisrun)

# + pycharm={"name": "#%%\n"}
fig = px.scatter(alldat, x="Principal component 1", y="Principal component 2", color='sensor', opacity=0.5)
fig.show()


# +



















