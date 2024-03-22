import sys
sys.path.append('/home/gluo/mitax')

import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'

from mitax.sampler import AncestralSampler
from mitax.misc import utils

import jax.numpy as jnp
import jax
import numpy as np

logdir='/home/gluo/mitax/examples/log/20240110-201054'
config = utils.load_config(logdir+'/config.yaml')
savecfl  = lambda p, a: utils.writecfl(p, utils.float2cplx(a)) if a.shape[-1] == 2 else utils.writecfl(p, a)
evalpath = lambda pr, ch: os.path.join(pr, ch)

sampler = AncestralSampler(net_name      = config['net_name'],
                           net_hparams   = config['net_hparams'],
                           dm_name       = config['loss_name'],
                           dm_hparams    = config['loss_params'],
                           init_input    = {'x': jnp.ones((1, 256, 256, 2), dtype=jnp.float32), 't': jnp.ones((1), dtype=jnp.float32)},
                           path          = logdir+'/mitax.model.unet.ncsnpp_919')


input_shape=(10, 256, 256, 2)
sampler.dm.sigma_max = 10.
sampler.dm.sigma_min = 0.005
sampler.dm.continuous = True
sampler.dm.N = 50
sampler.create_functions()
image1 = sampler(jax.random.normal(jax.random.PRNGKey(4), input_shape)*sampler.dm.sigma_max, inner_steps=1)

sampler.dm.continuous = False
sampler.create_functions()
image2 = sampler(jax.random.normal(jax.random.PRNGKey(4), input_shape)*sampler.dm.sigma_max, inner_steps=1)

savecfl(evalpath('./', 'an'), np.concatenate((image1[-1], image2[-1]), axis=0))
