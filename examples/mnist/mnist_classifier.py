
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='True'

import sys
sys.path.append('/home/gluo/mitax')

from mitax.trainer import trainer
from mitax.misc import utils
import jax.numpy as jnp
import numpy as np

from mnist import Mnist

def map_f(x):
    return x[0][..., np.newaxis], x[1]

config = utils.load_config('mnist.yaml')

d1 = Mnist('train', True, './')
train_pipe = utils.dataloader(d1, config['num_thread'], map_f, config['batch_size'], strict=True)

d2 = Mnist('test', True, './')
test_pipe = utils.dataloader(d2, config['num_thread'], map_f, config['batch_size'], strict=True)

tx = trainer(net_name    = config['net_name'],
             net_hparams = {},
             loss_name     = config['loss_name'],
             loss_params  = {},
             optimizer_name = config['optimizer'],
             optimizer_hparams =config['optimizer_hparams'], 
             init_input        = {'x': jnp.ones([1, 28, 28, 1])},
             logdir = config['logdir'])

tx.train(train_pipe, test_pipe, 30, len(train_pipe), 10)