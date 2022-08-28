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

# ### Installation
#     
# requirement.txt can be found in the repository. In order install all necessary packages you should run:
#
# pip install -r requirements.txt
#
# Adding Calina library to pythonpath is also necessary. If you are using using anaconda you can just type:
#
# conda develop Calina_path

# ### Importing required modules

import pandas as pd
# %load_ext autoreload
# %autoreload 2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ### Loading data into dataset

# +

from calina_dataset.calibration_dataset import Tell1Dataset, get_module_map

class MyDS(Tell1Dataset):
    filename_format = '%Y-%m-%d'
    filename_regex_format = r'\d{4}-\d{2}-\d{2}.csv'

datapath = "../../data/calibrations/"
data_list = MyDS.get_filepaths_from_dir(datapath)
mds = MyDS(data_list, read=True)

# -

modmap = get_module_map()

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

DATA_ = data.copy()

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
    #data_transponsed = data.transpose()
    data_transponsed = data.copy()
    pca = PCA(n_components = percent,svd_solver = 'auto')
    result = pca.fit_transform(data_transponsed)
    column_names = [f"Principal component {x+1}" for x in range(len(result[0]))]
    row_names = [f"Channel {x}" for x in range(len(result))]
    return pd.DataFrame(result,columns = column_names,index = row_names)


def full_pca_trans(data,percent):
    data = StandardScaler().fit_transform(data)
    data_transponsed = data.transpose()
    #data_transponsed = data.copy()
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
        print(sensor_data_key, dataset.shape)
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



def plotallped(t, trans=False):
    d = DATA_['pedestal'][t]
    single_data = {k: v.drop('sensor',axis=1) for k, v in d.groupby('sensor')}
    alpha = 0.4
    thisrun = []
    for sensor_data_key in single_data:
        dataset = single_data[sensor_data_key]
        dataset_after_pca = full_pca(dataset,procentage_or_num_of_comp) if not trans else full_pca_trans(dataset, procentage_or_num_of_comp)
        dataset_after_pca["sensor"] = str(int(sensor_data_key))
        #scatter = plt.scatter(dataset_after_pca.iloc[:,0], dataset_after_pca.iloc[:,1], edgecolor='none', alpha=alpha,label=sensor_data_key)
        #plt.legend(title="Module nr.")
        thisrun.append(dataset_after_pca)
    alldat = pd.concat(thisrun)
    fig = px.scatter(alldat, x="Principal component 1", y="Principal component 2", color='sensor', opacity=0.5)
    fig.show(renderer="notebook") 
    fig.write_html("PCA_{}.html".format(t))
    
    return alldat, fig


alldat, _ = plotallped('phi')

# + pycharm={"name": "#%%\n"}
alldat, _ = plotallped('r_phi')
# -


tmpdat = mds.dfp.df
traindat = tmpdat[tmpdat['sensor_type']=='phi'].iloc[:,9:]

traindat['sensor'] = tmpdat['sensor']
traindat['timeplay'] = tmpdat.datetime.astype(str)

gd = traindat.copy()
gd.groupby('sensor').transform(lambda x: x.T)
gd_sensor = gd[['sensor', 'timeplay']]
del gd['sensor']
del gd['timeplay']

dmf = traindat.drop(['timeplay'], axis=1)


data = StandardScaler().fit_transform(gd)
data_transponsed = data
pca = PCA(n_components = 2,svd_solver = 'auto')
result = pca.fit_transform(data_transponsed)


newdat = pd.DataFrame({'sensor': gd_sensor['sensor'], 'timeplay':gd_sensor['timeplay'], 'a':result[:,0], 'b':result[:,1]})



# + pycharm={"name": "#%%\n"}
fig = px.scatter(newdat, x="a", y="b", color='sensor', opacity=0.5)
full_fig = fig.full_figure_for_development()
fig.show(renderer="notebook") 
fig.write_html("PCA.html")


# + pycharm={"name": "#%%\n"}
fig = px.scatter(newdat, x="a", y="b", color='sensor', animation_frame='timeplay', opacity=0.5)
fig.update_layout(xaxis=dict(range=full_fig.layout.xaxis.range),yaxis=dict(range=full_fig.layout.yaxis.range))
fig.show(renderer="notebook") 
fig.write_html("PCA.html")
# -


mydata = mds.dfh.df


mydata.iloc[:,9:]

data1 = StandardScaler().fit_transform(mydata.iloc[:,9:])
pca = PCA(n_components = 2,svd_solver = 'auto')
result = pca.fit_transform(data1)

myd = mydata.iloc[:,:9]
myd['timeplay'] = myd.datetime.astype(str)

myd['A']=result[:, 0]
myd['B']=result[:, 1]

myd[myd['sensor_type']=='R']

badind = ((myd.datetime == '2012-07-30') | (myd.datetime == '2012-08-01'))
myd['symbol'] = 'other'
myd['symbol'][badind] = 'anomaly'



fig = px.scatter(myd[myd['sensor_type']=='R'], x="A", y="B", color='sensor', opacity=0.5, symbol='symbol', symbol_sequence=symseq)
fig.update_layout(legend=dict(
yanchor="top",
y=0.99,
xanchor="left",
x=0.01))
full_fig = fig.full_figure_for_development()
fig.show(renderer="notebook") 
fig.write_html("PCA.html")
fig.write_image("pics/PCA_module_R_all.png")

fig = px.scatter(myd[myd['sensor_type']=='phi'], x="A", y="B", color='sensor', opacity=0.5,  symbol='symbol', symbol_sequence=symseq)
fig.update_layout(legend=dict(
yanchor="top",
y=0.99,
xanchor="left",
x=0.01))
full_fig = fig.full_figure_for_development()
fig.show(renderer="notebook") 
fig.write_html("PCA.html")
fig.write_image("pics/PCA_module_phi_all.png")

full_fig.layout.xaxis.range

fig = px.scatter(myd[myd['sensor_type']=='R'], x="A", y="B", color='sensor', opacity=0.5, animation_frame="timeplay")
fig.update_layout(xaxis=dict(range=full_fig.layout.xaxis.range),yaxis=dict(range=full_fig.layout.yaxis.range))
fig.show(renderer="notebook") 
fig.write_html("PCA.html")


def getanyplot(t):
    return lambda x: px.scatter(myd[(myd['sensor_type']==t) & (myd['timeplay']==x)], x="A", y="B", color='sensor', opacity=0.5,)
getplotR = getanyplot('R')
getplotphi = getanyplot('phi')

# +
plotdates = [
    '2012-04-27',
    '2012-05-04',
'2012-06-27',
'2012-07-06',
'2012-07-30',
'2012-08-01',
'2012-08-02',
'2012-08-14',
    '2012-09-23',
    '2013-01-30'
    ]


from plotly.subplots import make_subplots

def maketogetherplot(t):
    fig2 = px.scatter(myd[myd['sensor_type']==t], x="A", y="B", color='sensor', opacity=0.5)
    full_fig = fig2.full_figure_for_development()
    fig = make_subplots(rows=2, cols=5, shared_xaxes=True,  shared_yaxes=True, subplot_titles=plotdates)

    for i, dat in enumerate(plotdates):
        trc = getanyplot(t)(dat)
        # fig.update_layout(xaxis=dict(range=full_fig.layout.xaxis.range),yaxis=dict(range=full_fig.layout.yaxis.range))
        # fig.write_image("pics/NN_module_{}_{}.png".format(stype, i))

        x = (i%5)+1
        y = (i//5)+1
        # trc = getplotR(date)
        trc.update_layout(xaxis=dict(range=full_fig.layout.xaxis.range),yaxis=dict(range=full_fig.layout.yaxis.range), yaxis_title=dat)
        fig.append_trace(trc['data'][0],col=x, row=y)
        fig['layout']['yaxis{}'.format(i+1)]['range']=full_fig.layout.yaxis.range
        fig['layout']['xaxis{}'.format(i+1)]['range']=full_fig.layout.xaxis.range
    fig.update_layout( autosize=False, width=1200, height=600,)

    fig.write_image("pics/PCA_module_{}_together.png".format(t))
    fig.show()


# -

maketogetherplot("R")

maketogetherplot("phi")

# +
from plotly.subplots import make_subplots
fig = make_subplots(rows=5, cols=1, shared_xaxes=True)

date = '2012-07-06'
fig['layout']['yaxis1']['title']=date
trc = getplotR(date)
trc.update_layout(xaxis=dict(range=full_fig.layout.xaxis.range),yaxis=dict(range=full_fig.layout.yaxis.range), yaxis_title=date)
fig.append_trace(trc['data'][0],
    row=1, col=1)


date = '2012-07-30'
fig['layout']['yaxis2']['title']=date
trc = getplotR(date)
trc.update_layout(xaxis=dict(range=full_fig.layout.xaxis.range),yaxis=dict(range=full_fig.layout.yaxis.range), yaxis_title=date)
fig.append_trace(trc['data'][0],
    row=2, col=1)

date = '2012-08-01'
fig['layout']['yaxis3']['title']=date
trc = getplotR(date)
trc.update_layout(xaxis=dict(range=full_fig.layout.xaxis.range),yaxis=dict(range=full_fig.layout.yaxis.range), yaxis_title=date)
fig.append_trace(trc['data'][0],
    row=3, col=1)

date = '2012-08-02'
fig['layout']['yaxis4']['title']=date
trc = getplotR(date)
trc.update_layout(xaxis=dict(range=full_fig.layout.xaxis.range),yaxis=dict(range=full_fig.layout.yaxis.range), yaxis_title=date)
fig.append_trace(trc['data'][0],
    row=4, col=1)

date = '2012-08-14'
trc = getplotR(date)
trc.update_layout(xaxis=dict(range=full_fig.layout.xaxis.range),yaxis=dict(range=full_fig.layout.yaxis.range), yaxis_title=date)
fig.append_trace(trc['data'][0],
    row=5, col=1)
fig.update_layout( autosize=False, width=400, height=1200,)


fig.update_layout({
               'xaxis1':{'range': full_fig.layout.xaxis.range},
               'yaxis1':{'range': full_fig.layout.yaxis.range},

               'xaxis2':{'range': full_fig.layout.xaxis.range},
               'yaxis2':{'range': full_fig.layout.yaxis.range},

               'xaxis3':{'range': full_fig.layout.xaxis.range},
               'yaxis3':{'range': full_fig.layout.yaxis.range},

               'xaxis4':{'range': full_fig.layout.xaxis.range},
               'yaxis4':{'range': full_fig.layout.yaxis.range},

               'xaxis5':{'range': full_fig.layout.xaxis.range},
               'yaxis5':{'range': full_fig.layout.yaxis.range},
}
)



fig['layout']['yaxis5']['title']=date
fig.write_image("pics/PCA_module_R_together.png")
fig.show()



# +


fig = getplotR('2012-07-30')
fig.update_layout(xaxis=dict(range=full_fig.layout.xaxis.range),yaxis=dict(range=full_fig.layout.yaxis.range))
fig.write_image("pics/PCA_module_R_2.png")

fig = getplotR('2012-08-01')
fig.update_layout(xaxis=dict(range=full_fig.layout.xaxis.range),yaxis=dict(range=full_fig.layout.yaxis.range))
fig.write_image("pics/PCA_module_R_3.png")

fig = getplotR('2012-08-02')
fig.update_layout(xaxis=dict(range=full_fig.layout.xaxis.range),yaxis=dict(range=full_fig.layout.yaxis.range))
fig.write_image("pics/PCA_module_R_4.png")

fig = getplotR('2012-08-14')
fig.update_layout(xaxis=dict(range=full_fig.layout.xaxis.range),yaxis=dict(range=full_fig.layout.yaxis.range))
fig.write_image("pics/PCA_module_R_5.png")


fig.write_image("pics/PCA_module_R_1.png")
# -



symseq = ['circle', 'x']
fig = px.scatter(myd[myd['sensor_type']=='phi'], x="A", y="B", color='sensor', opacity=0.5, symbol='symbol', symbol_sequence=symseq)
fig.update_layout(legend=dict(
yanchor="top",
y=0.99,
xanchor="left",
x=0.01))
full_fig = fig.full_figure_for_development()
fig.show(renderer="notebook") 
fig.write_html("PCA.html")
fig.write_image("pics/PCA_module_phi_all.png")

fig = px.scatter(myd[myd['sensor_type']=='phi'], x="A", y="B", color='sensor', opacity=0.5, animation_frame="timeplay")
fig.update_layout(xaxis=dict(range=full_fig.layout.xaxis.range),yaxis=dict(range=full_fig.layout.yaxis.range))
fig.show(renderer="notebook") 
fig.write_html("PCA.html")



datype = 'r_phi'
alldat, figall = plotallped(datype, True)
part1 = alldat.sensor.unique()[:11]
part2 = alldat.sensor.unique()[11:21]
part3 = alldat.sensor.unique()[21:32]
part4 = alldat.sensor.unique()[32:]
alldat1 = alldat[alldat.sensor.isin(part1)]
alldat2 = alldat[alldat.sensor.isin(part2)]
alldat3 = alldat[alldat.sensor.isin(part3)]
alldat4 = alldat[alldat.sensor.isin(part4)]
figall.write_image("pics/PCA_pedestals_all_{}.png".format(datype))



def minmax(df):
    ran=  max(df) - min(df)
    half = ran/2
    mid = min(df) + half
    scale = 1.05
    min_, max_ = mid - scale*half, mid + scale*half
    return min_, max_


xaxisrange = dict(range=minmax(alldat["Principal component 1"]))
yaxisrange = dict(range=minmax(alldat["Principal component 2"]))


# + pycharm={"name": "#%%\n"}
def produce2plot(data, xaxisrange, yaxisrange):
    from plotly.graph_objects import Layout

    fig = px.scatter(data, x="Principal component 1", y="Principal component 2", color='sensor', opacity=0.5, width=600, height=600)
    fig.update_xaxes(
        title_font=dict(size=15, family='Courier', color='black'),
        showgrid=True,
        gridwidth=1,
        gridcolor='gray',
        zerolinewidth=1,
        zerolinecolor='gray',
        linecolor='black',
        mirror=True,
    )
    fig.update_yaxes(
        title_font=dict(size=15, family='Courier', color='black'),         
        scaleanchor = "x",
        scaleratio = 1,
        showgrid=True,
        gridwidth=1,
        gridcolor='gray',
        zerolinewidth=1,
        zerolinecolor='gray',
        linecolor='black',
        mirror=True,
    )
    fig.update_layout(xaxis=xaxisrange, yaxis=yaxisrange, plot_bgcolor='rgba(0,0,0,0)')
    
    #fig.show(renderer="notebook") 
    #fig.write_image("pics/PCA_pedestals.png")
    return fig


# -

dats = [alldat1, alldat2, alldat3, alldat4]
for i, d in enumerate(dats):
    fig = produce2plot(d, xaxisrange, yaxisrange)
    fig.write_image('pics/PCA_pedestals_{}_{}.png'.format(datype, i))

selected_points = ['Channel 543','Channel 1899','Channel 322']

alldat11.iloc[0,3]



# + pycharm={"name": "#%%\n"}
alldat11 = alldat2.copy()
alldat11 = alldat2[alldat2.sensor == '11']
alldat11.loc[selected_points, 'sensor'] = 'other'
alldat11['alpha'] = 0.3
alldat11.loc[selected_points, 'alpha'] = 1.0
#fig.clear()
fig = px.scatter(alldat11[alldat11.sensor == '11'], x="Principal component 1", y="Principal component 2", color='sensor', opacity=0.3, hover_name=alldat11[alldat11.sensor=='11'].index, width=600, height=600)
# -

alldsel.loc['Channel 543']

fig, axe = plt.subplots(1,1, figsize=(6,6))
axe.scatter(x=alldat11["Principal component 1"], y=alldat11["Principal component 2"], color='blue')
alldsel = alldat11[alldat11.sensor == 'other']
axe.scatter(x=alldsel["Principal component 1"], y=alldsel["Principal component 2"], color='red')
for i, txt in enumerate(selected_points):
    x,y = (alldsel.loc[txt, "Principal component 1"], alldsel.loc[txt,"Principal component 2"])
    dx, dy = [(-1.5,2),(1.2,1.5),(0,-1.2)][i]
    axe.annotate(txt, 
                 (x,y),
                 (x+dx,y+dy),
                arrowprops=dict(arrowstyle="->", color='y', lw=3))
axe.grid()

fig, axe = plt.subplots(1,1,figsize=(6,3))
for channel in selected_points:
    stmpd[channel.lower().replace(' ', '')].plot(ax=axe)
axe.legend()
axe.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16),
          ncol=3, fancybox=True, shadow=True)
fig.savefig("pics/PCA_trends_channel.png")






