"""
This module was made to convert the networks that use the pytorch lightning with the old version, to purely torch
dependant modules.

This needs to be runned using pytorch_lightning == 1.0.8
"""

import torch
import os
import neptune.new as neptune
import zipfile
import pytorch_lightning as pl

import pathlib


def convert_network(exp_id):
    run = neptune.init('pawel-drabczyk/velodimred', run=exp_id, mode="read-only", capture_stderr=False, capture_stdout=False)

    directory = pathlib.Path('legacy_network')
    directory.mkdir(parents=True, exist_ok=True)
    run['artifacts/trained_model.ckpt'].download(str(directory))
    run['source_code/files'].download(str(directory))
    with zipfile.ZipFile("legacy_network/files.zip","r") as zip_ref:
            zip_ref.extractall('legacy_network')
    # (directory/"__init__.py").touch()
    # uncomment the following line when the dependancy injection problem is solved
    # from legacy_network.networks import VeloDecoder, VeloEncoder, VeloAutoencoderLt
    import calina_dataset.calibration_dataset as calibration_dataset
    import sys
    sys.modules['calibration_dataset'] = calibration_dataset
    from legacy_network.networks import VeloDecoder, VeloEncoder
    from legacy_network import networks
    sys.modules['networks'] = networks
    from network import VeloAutoencoder
    model = torch.load(directory/'trained_model.ckpt', map_location=torch.device('cpu'))
    new_model = VeloAutoencoder(model.enc, model.dec)
    torch.save(new_model, 'torch_only_model.ckpt')
    run_to_save = neptune.init('mmajewsk/pubdimred', tags=['conversion', str(exp_id)], source_files=['network.py'])
    run_to_save['artifacts'].upload('torch_only_model.ckpt')




def read_as_new_network(path):
    from network import VeloAutoencoder, VeloDecoder, VeloEncoder
    return torch.load(path)


if __name__ == "__main__":
    convert_network('VEL-371')
