from typing import Any, Callable, Optional
import jax.numpy as jnp
import orbax.checkpoint as ocp
from functools import partial

import jax
import tqdm

from mitax.misc import utils

class base():
    """
    Base class for the sampler.

    Args:
        net_name (str): Name of the network.
        net_hparams (dict): Hyperparameters for the network.
        dm_name (str): Name of the diffusion model.
        dm_hparams (dict): Hyperparameters for the diffusion model.
        target_snr (float): Target signal-to-noise ratio.
        cond_func (callable): Conditioning function.
        seed (int): Random seed.

    Attributes:
        net_name (str): Name of the network.
        net_hparams (dict): Hyperparameters for the network.
        dm_name (str): Name of the diffusion model.
        dm_hparams (dict): Hyperparameters for the diffusion model.
        target_snr (float): Target signal-to-noise ratio.
        cond_func (callable): Conditioning function.
        seed (int): Random seed.
        orbax_checkpointer (PyTreeCheckpointer): Checkpointer for saving and restoring model weights.
        net_weights (dict): Restored network weights.
        dm (DiffusionModel): Initialized diffusion model.
        net (Network): Initialized network.
    """

    def __init__(self,
                net_name: str,
                net_hparams: dict,
                dm_name: str,
                dm_hparams: dict,
                target_snr: float,
                cond_func: Optional[Callable],
                rng_key=jax.random.PRNGKey(0),
                ):
        """
        Initializes the base sampler class with the given parameters.
        """
        self.net_name      = net_name
        self.net_hparams   = net_hparams
        self.dm_name       = dm_name
        self.dm_hparams    = dm_hparams
        self.target_snr    = target_snr
        self.cond_func     = cond_func
        self.rng_key       = rng_key
        self.orbax_checkpointer = ocp.PyTreeCheckpointer()
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
            raise NotImplementedError
    
    def create_functions(self):
        """
        Creates the functions used for sampling when initializing and updating the params.

        """
        raise NotImplementedError

    def load_weights(self, path):
        """
        Restores the training state from a checkpoint file.

        Args:
            path (str): Path to the checkpoint file.

        Returns:
            None
        Raises:
            ValueError: If failed to load the model.
        """
        try:
            self.net_weights = self.orbax_checkpointer.restore(path)['model']['params']
        except:
            raise ValueError('Failed to load model')

    def init_dm(self, params):
        """
        Initializes the diffusion model with the given parameters.

        Args:
            params (dict): A dictionary of parameters to be passed to the diffusion model constructor.

        Returns:
            None
        """

        module_name, class_name = self.dm_name.rsplit('.', 1)
        self.dm = utils.get_class_by_name(module_name, class_name)(**params)

    def init_model(self, init_input, path):
        """
        Initializes the model with the given name, parameters, and input.

        Args:
            init_input: The input used for initializing the model.
            path (str): Path to the weights file.

        Returns:
            None
        """

        module_name, class_name = self.net_name.rsplit('.', 1)
        self.net = utils.get_class_by_name(module_name, class_name)(**self.net_hparams)
        _  = self.net.init(self.rng_key, **init_input, train=False)
        self.init_dm(self.dm_hparams)
        self.load_weights(path)

class AncestralSampler(base):

    def __init__(self, net_name, net_hparams, dm_name, dm_hparams, target_snr, path, init_input, cond_func=None, rng_key=jax.random.PRNGKey(0)):
        super().__init__(net_name, net_hparams, dm_name, dm_hparams, target_snr, cond_func, rng_key)
        self.init_model(init_input, path)
        self.create_functions()


    def create_functions(self):

        self.dm.set_d_step(self.dm.N)

        def update_step(x, t, key):
            """
            Performs an update step.

            Args:
                x (Tensor): Input tensor.
                t (Tensor): Time tensor.

            Returns:
                Tensor: Updated input tensor.
            """
            if self.dm.continuous:
                reverse = self.dm.reverse_sde
            else:
                reverse = self.dm.reverse_discrete

            drift, diffusion = reverse(self.net.apply, self.net_weights, x, t)
            
            x_mean = x - drift
            x      = x_mean + jax.random.normal(key, jnp.shape(x))*diffusion
            return x, x_mean
        
        self.update_step = jax.jit(update_step)

    def __call__(self, x_init, inner_steps=1):
        """
        Perform the sampling process.

        Args:
            x_init (numpy.ndarray): The initial value of x.
            inner_steps (int): The number of inner steps to perform.

        Returns:
            tuple: A tuple containing two lists: xs and xs_mean.
                - xs (list): A list of sampled values of x.
                - xs_mean (list): A list of mean values of x.

        """
        x_val     = x_init

        xs      = []
        xs_mean = []

        nr_samples = jnp.shape(x_init)[0]

        for t_i in tqdm.tqdm(jnp.linspace(self.dm.T, self.dm.eps, self.dm.N)):
            for _ in range(inner_steps):
                self.rng_key, subkey = jax.random.split(self.rng_key)
                x_val, x_mean = self.update_step(x_val, jnp.array([t_i for _ in range(nr_samples)]), subkey)
                
                if self.cond_func is not None:
                    pass # Not implemented yet

            xs.append(x_val)
            xs_mean.append(x_mean)

        return xs, xs_mean
    
def progressive_sampling(sampler, Ns, sigmaxs, rhos, x_init, inner_steps=1):

    k = len(Ns)
    x_val = x_init

    for i in range(k):

        sampler.dm.sigma_max = sigmaxs[i]
        sampler.dm.rho       = rhos[i]
        sampler.dm.set_d_step(Ns[i])
        sampler.create_functions()
        _, image = sampler(x_val, inner_steps)
        x_val    = image[-1]

    return image