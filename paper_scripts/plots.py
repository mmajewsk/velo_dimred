
import neptune.new as neptune
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

if __name__ == '__main__':
    runid = 'PUBDIM-31'
    projectname = 'mmajewsk/pubdimred'
    run = neptune.init(projectname, run=runid, mode="read-only", capture_stderr=False, capture_stdout=False)
    loss = run['logs/loss'].fetch_values()
    val_loss = run['logs/val_loss'].fetch_values()
    ax = loss.plot(x='step', y='value', label='loss', ylabel='mse')
    val_loss.plot(x='step', y='value', ax=ax, label='validation loss')
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end, 0.05))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    ax.grid()
    fig = ax.get_figure()
    fig.savefig('training_plot.png')

    # breakpoint()
