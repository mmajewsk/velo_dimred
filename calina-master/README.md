# calina - Calibration Intelligence and Analysis

This project was created with *python 3* to find anomalies and analyse VeLo calibration. 
There are two main functionalities:
1. Process and analyse data
2. Save the results to database

## Installation and dependencies

As for now, the parameters of the calibration models are constant and saved in `model_parameters`.
The learning script will be added later.

The main dependancy is the [dataset](https://dataset.readthedocs.io/en/latest/) package - 
its used for database manipulation.
So installing dependencies should be as simple as running `pip install dataset`.

### On lxplus
As far as i know lxplus has no python3 installed, so you might wanna install (miniconda)[https://conda.io/miniconda.html.]
Lxplus has access to the internet so you just have to install pip if you don't have one.


### On plus
This one is tricky, you need to manually download packages and pip on some other machine, and then move them to plus.
So first, you need pip. Im using the method described [here](https://github.com/pypa/pip/issues/2351#issuecomment-69994524).
First download pip wheels from [pypi](https://pypi.org/project/pip/#files).
And move them to plus, then on plus run.
```
python pip-6.0.6-py2.py3-none-any.whl/pip install --no-index pip-6.0.6-py2.py3-none-any.whl --target=/destination/for/pip

```
(change the package version and destination accordingly)

Ok, but you still need dataset, but there is no internet access on plus so you 
need again manually download packages and move them to plus, as described [here](https://stackoverflow.com/a/14447068).
On a machine **other than plus** do:
```
pip install --download /path/to/some/dir dataset

```

Then you should see all the dependencies downloaded to selected folder.
Now **move them to plus and using the installed pip do**:
```
./destination/for/pip/bin/pip install --no-index --target=/destination/for/dataset --find-links /path/to/some/dir/ dataset
```
Remember to change the `--target=/destination/for/dataset` accordingly.

Next, before you run this project you need to add the `/destination/for/dataset` to PYTHONPATH, like this:
```
export PYTHONPATH=/destination/for/dataset:$PYTHONPATH
```

and that should be it!

##Usage
```
usage: calina.py [-h] [-d DESTINATION] [--run-list] [--force_recalculation]
                 source

" Calculates the outlierness for calibration of VeLo. By default it will not
recalculate existing callibrations (those that exist in 'by_day' table in
database.

positional arguments:
  source                path to calibration database dump directory or runlist

optional arguments:
  -h, --help            show this help message and exit
  -d DESTINATION        path to database
  --run-list            source is path to RunList.txt (ending with
                        'RunList.txt')
  --force_recalculation
                        force recalculation of the callibrations that already
                        exists in calina database
```
