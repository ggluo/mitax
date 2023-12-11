from mitax.trainer import trainer
from mitax.misc import utils

from mnist import Mnist
import jax.numpy as jnp
import numpy as np

import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='3'


def map_f(x):
    return x[0][..., np.newaxis], x[1]

config = utils.load_config('./mnist.yaml')

d1 = Mnist('train', True, './')
train_pipe = utils.dataloader(d1, config['num_thread'], map_f, config['batch_size'], strict=True)

d2 = Mnist('test', True, './')
test_pipe = utils.dataloader(d2, config['num_thread'], map_f, config['batch_size'], strict=True)

tx = trainer(model_name    = config['model_name'],
             model_hparams = {},
             loss_name     = config['loss_name'],
             loss_params  = {},
             optimizer_name = config['optimizer'],
             optimizer_hparams ={'lr': config['lr']}, 
             init_input        = jnp.ones([1, 28, 28, 1]),
             logdir = config['logdir'])

tx.train(train_pipe, test_pipe, 10, len(train_pipe), 1)