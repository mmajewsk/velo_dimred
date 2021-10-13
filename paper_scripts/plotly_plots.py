import pandas as pd
from sklearn import preprocessing
import os
import plotly.express as px
import neptune.new as neptune
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.ticker as ticker
import zipfile
from calina_dataset.calibration_dataset import Tell1Dataset
    # breakpoint()


class MyDS(Tell1Dataset):
    filename_format = '%Y-%m-%d'
    filename_regex_format = r'\d{4}-\d{2}-\d{2}.csv'

def slider_plot(model, run, stype):
    # indexesList = metadata.index.values.tolist()
    # xyDF = pd.DataFrame(reducedData, index=indexesList, columns=['x', 'y'])
    # resultDF = pd.concat([metadata, xyDF], axis=1)
    # resultDF["datetime"] = resultDF["datetime"].astype(str)
    # fig = px.scatter(resultDF, x="x", y="y", animation_frame="datetime", animation_group="sensor", color="sensor")
    # fig["layout"].pop("updatemenus") # optional, drop animation buttons
    # fig.update_xaxes(range=[1.15*resultDF['x'].min(), 1.15*resultDF['y'].max()])
    # fig.update_yaxes(range=[1.15*resultDF['x'].min(), 1.15*resultDF['y'].max()])

    # dfh_r = pd.DataFrame(dfh_r_scaled, index=dfh_r.index, columns=dfh_r.columns)


    scaler = preprocessing.StandardScaler()

    datapath = os.path.join("../../../data", "calibrations")
    data_list = MyDS.get_filepaths_from_dir(datapath)
    mds = MyDS(data_list, read=True)
    dfh_r = mds.dfh[stype].df.iloc[:, 9:]
    dfh_r_scaled = scaler.fit_transform(dfh_r)
    # dfh_r = pd.DataFrame(dfh_r_scaled, index=dfh_r.index, columns=dfh_r.columns)

    reducedData = model.enc.forward(torch.tensor(dfh_r_scaled, dtype=torch.float))
    reducedData = reducedData.detach().numpy()
    rdd = mds.dfh[stype].df[['sensor', 'datetime']]
    rdd.datetime = rdd.datetime.astype('str')
    rdd['A'] = reducedData[:,0]
    rdd['B'] = reducedData[:,1]
    fig = px.scatter(rdd, x="A", y="B", color='sensor', opacity=0.5)
    full_fig = fig.full_figure_for_development()
    fig.show(renderer="notebook")
    fig.write_html("PCA.html")
    fig.write_image("pics/NN_module_{}_all.png".format(stype))

    # breakpoint()
    getplotR = lambda x: px.scatter(rdd[rdd['datetime']==x], x="A", y="B", color='sensor', opacity=0.5)

    plotdates = [
    '2012-07-06',
    '2012-07-30',
    '2012-08-01',
    '2012-08-02',
    '2012-08-14',
        ]
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True,  shared_yaxes=True)

    for i, dat in enumerate(plotdates):
        trc = getplotR(dat)
        # fig.update_layout(xaxis=dict(range=full_fig.layout.xaxis.range),yaxis=dict(range=full_fig.layout.yaxis.range))
        # fig.write_image("pics/NN_module_{}_{}.png".format(stype, i))

        # trc = getplotR(date)
        trc.update_layout(xaxis=dict(range=full_fig.layout.xaxis.range),yaxis=dict(range=full_fig.layout.yaxis.range), yaxis_title=dat)
        fig.append_trace(trc['data'][0],row=i+1, col=1)
        fig['layout']['yaxis{}'.format(i+1)]['title']=dat


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
    fig.write_image("pics/NN_module_{}_together.png".format(stype))
    return fig

def reopen_experiment(run):
    run['artifacts/trained_model.ckpt'].download()
    run['source_code/files'].download()
    with zipfile.ZipFile("files.zip","r") as zip_ref:
        zip_ref.extractall('legacy_network')

    import calina_dataset.calibration_dataset as calibration_dataset
    import sys
    sys.modules['calibration_dataset'] = calibration_dataset
    from legacy_network.networks import VeloDecoder, VeloEncoder, VeloAutoencoderLt
    import legacy_network.networks as networks
    sys.modules['networks'] = networks
    model = torch.load('trained_model.ckpt', map_location=torch.device('cpu'))
    return model

if __name__ == '__main__':
    runid = 'PUBDIM-46'
    projectname = 'mmajewsk/pubdimred'
    run = neptune.init(projectname, run=runid, mode="read-only", capture_stderr=False, capture_stdout=False)
    model = reopen_experiment(run)
    fig = slider_plot(model, run, 'phi')
    fig = slider_plot(model, run, 'R')
