
import neptune.new as neptune
import matplotlib.pyplot as plt

if __name__ == '__main__':
    runid = 'PUBDIM-31'
    projectname = 'mmajewsk/pubdimred'
    run = neptune.init(projectname, run=runid, mode="read-only", capture_stderr=False, capture_stdout=False)
    loss = run['logs/loss'].fetch_values()
    val_loss = run['logs/val_loss'].fetch_values()
    ax = loss.plot(x='step', y='value', label='loss')
    val_loss.plot(x='step', y='value', ax=ax, label='validation_loss')
    fig = ax.get_figure()
    fig.savefig('training_plot.png')

    # breakpoint()
