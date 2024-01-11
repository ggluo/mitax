import sys
sys.path.append('/home/gluo/mitax')

import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='2'

from mitax.trainer import trainer
from mitax.misc import utils
import numpy as np
import jax
import jax.numpy as jnp

def map_f(x):
    x = np.load(x)['rss']
    x = x / np.max(abs(x))
    return utils.cplx2float(x)


config = utils.load_config('/home/gluo/mitax/examples/smld_unet.yaml')

d1 = utils.fileflow(utils.list_files('/home/gluo/brain_mnist/train', ext='npz'), shuffle=True)
d2 = utils.fileflow(utils.list_files('/home/gluo/brain_mnist/test', ext='npz'), shuffle=False)

train_pipe = utils.dataloader(d1, config['num_thread'], map_f, config['batch_size'], strict=True)
test_pipe  = utils.dataloader(d2, config['num_thread'], map_f, config['batch_size'], strict=True)

init_x =next(iter(train_pipe))

tx = trainer(net_name          = config['net_name'],
             net_hparams       = config['net_hparams'],
             loss_name         = config['loss_name'],
             loss_params       = config['loss_params'],
             optimizer_name    = config['optimizer'],
             optimizer_hparams = config['opt_hparams'], 
             init_input        = {'x': init_x, 't': jnp.ones((config['batch_size']), dtype=jnp.float32)},
             logdir            = config['logdir'],
             rng_key           = jax.random.PRNGKey(0))
utils.save_config(config, tx.logdir)

tx.train(train_pipe, test_pipe, config['epoch'], len(train_pipe), 10)

