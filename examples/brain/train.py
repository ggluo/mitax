import os
import sys
sys.path.append('/home/gluo/mitax')

import jax
import jax.numpy as jnp

from mitax.trainer import trainer
from mitax.misc import utils
import numpy as np


def main(config=utils.load_config('/home/gluo/mitax/examples/brain/cm_smld.yaml'), gpu_id='3', train_files='/scratch/gluo/brain_mnist/train', test_files='/scratch/gluo/brain_mnist/test', ext='npz'):
    
    os.environ['CUDA_VISIBLE_DEVICES']=gpu_id
    
    d1 = utils.fileflow(utils.list_files(train_files, ext=ext), shuffle=True)
    d2 = utils.fileflow(utils.list_files(test_files, ext=ext), shuffle=False)

    def map_f(x):
        x = np.load(x)['rss']
        x = x / np.max(abs(x))
        return utils.cplx2float(x)

    train_pipe = utils.dataloader(d1, config['num_thread'], map_f, config['batch_size'], strict=True, factor=3)
    test_pipe  = utils.dataloader(d2, config['num_thread'], map_f, config['batch_size'], strict=True, factor=3)

    init_x =next(iter(train_pipe))

    tx = trainer(net_name         = config['net_name'],
                net_hparams       = config['net_hparams'],
                loss_name         = config['loss_name'],
                loss_params       = config['loss_params'],
                optimizer_name    = config['optimizer'],
                optimizer_hparams = config['opt_hparams'], 
                init_input        = {'x': init_x, 't': jnp.ones((config['batch_size']), dtype=jnp.float32)},
                logdir            = config['logdir'],
                rng_key           = jax.random.PRNGKey(0))
    utils.save_config(config, tx.logdir)

    if 'save_interval' in config.keys():
        save_interval = config['save_interval']
    else:
        save_interval = 10

    tx.train(train_pipe, test_pipe, config['epoch'], len(train_pipe), save_interval)

if __name__ == "__main__":
    main(utils.load_config(sys.argv[1]), sys.argv[2])