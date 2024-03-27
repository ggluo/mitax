from typing import Any, Callable, Optional
import jax.numpy as jnp
import orbax.checkpoint as ocp
from functools import partial

import jax
import tqdm
import einops

from mitax.misc import utils

class base():
    """
    Base class for the sampler.

    Args:
        net_name (str): Name of the network.
        net_hparams (dict): Hyperparameters for the network.
        dm_name (str): Name of the diffusion model.
        dm_hparams (dict): Hyperparameters for the diffusion model.
        cond_func (callable): Conditioning function.
        seed (int): Random seed.

    Attributes:
        net_name (str): Name of the network.
        net_hparams (dict): Hyperparameters for the network.
        dm_name (str): Name of the diffusion model.
        dm_hparams (dict): Hyperparameters for the diffusion model.
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
                cond_func: Optional[Callable] = None,
                rng_key: Optional[Any] = None,
                ):
        """
        Initializes the base sampler class with the given parameters.
        """
        self.net_name      = net_name
        self.net_hparams   = net_hparams
        self.dm_name       = dm_name
        self.dm_hparams    = dm_hparams
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
        if self.rng_key is None:
            self.rng_key = jax.random.PRNGKey(0)

        module_name, class_name = self.net_name.rsplit('.', 1)
        self.net = utils.get_class_by_name(module_name, class_name)(**self.net_hparams)
        _  = self.net.init(self.rng_key, **init_input, train=False)
        self.init_dm(self.dm_hparams)
        self.load_weights(path)


#######################################################
#  - Sampler for the ancestral sampling process       #
#  - it discretizes the SDE and samples from it       #
#  - in an ancestral way and uses the Euler-Maruyama  #
#  - method at each transitional stage                #
#######################################################

class AncestralSampler(base):

    def __init__(self, net_name, net_hparams, dm_name, dm_hparams, path, init_input, cond_func=None, rng_key=None):
        super().__init__(net_name, net_hparams, dm_name, dm_hparams, cond_func, rng_key)
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
            x      = x_mean + jax.random.normal(jax.random.split(key)[0], jnp.shape(x))*diffusion
            return x, jnp.squeeze(diffusion)
        
        self.update_step = jax.jit(update_step)

    def __call__(self, x_init, inner_steps=1, save_evol=False):
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

        if save_evol:
            xs      = []

        nr_samples = jnp.shape(x_init)[0]

        for t_i in tqdm.tqdm(jnp.linspace(self.dm.T, self.dm.eps, self.dm.N)):
            for _ in range(inner_steps):
                self.rng_key, subkey = jax.random.split(self.rng_key)
                x_val, sig = self.update_step(x_val, jnp.array([t_i for _ in range(nr_samples)]), subkey)
                
                if self.cond_func is not None:
                    x_val = self.cond_func(x_val, sig)

            if save_evol:
                xs.append(x_val)

        if save_evol:
            return xs
        else:
            return x_val

#######################################################
# - PC sampler for the 2D data generation process     #
#######################################################
class PredCorrSampler(base):
    """
    """
    def __init__(self, net_name, net_hparams, dm_name, dm_hparams, path, init_input, cond_func=None, rng_key=None):
        super().__init__(net_name, net_hparams, dm_name, dm_hparams, cond_func, rng_key)
        self.init_model(init_input, path)
        self.create_functions()

    def create_functions(self, target_psnr=0.2):

        self.dm.set_d_step(self.dm.N)

        def predictor(x, t, k):
            if self.dm.continuous:
                reverse = self.dm.reverse_sde
            else:
                reverse = self.dm.reverse_discrete
            drift, diffusion  = reverse(self.net.apply, self.net_weights, x, t)
            z       = jax.random.normal(k, jnp.shape(x))
            x    = x - drift  + diffusion * z
            return x

        def norm(x):
            x_squared = jnp.sum(x**2, axis=(-3, -2, -1))
            x_l2 = jnp.sqrt(x_squared)
            x_l2_keepdims = jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(x_l2, -1), -1), -1)
            return x_l2_keepdims
    
        def corrector(x, t, k):
            grad       = self.dm.score(self.net.apply, self.net_weights, x, t)
            noise      = jax.random.normal(k, jnp.shape(x))
            grad_norm  = norm(grad)
            noise_norm = norm(noise)
            step_size  = (target_psnr*noise_norm / grad_norm) ** 2 * 2 * 1
            x          = x + step_size * grad + noise * jnp.sqrt(step_size * 2)
            return x
        
        def update_step(x, t, key):
            
            key, subkey = jax.random.split(key)
            x = corrector(x, t, subkey)
            
            key, subkey = jax.random.split(key)
            x = predictor(x, t, subkey)

            return x, None

        self.update_step = jax.jit(update_step)
    
    def __call__(self, x_init, inner_steps=1, save_evol=False):
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

        if save_evol:
            xs      = []

        nr_samples = jnp.shape(x_init)[0]

        for t_i in tqdm.tqdm(jnp.linspace(self.dm.T, self.dm.eps, self.dm.N)):
            for _ in range(inner_steps):
                self.rng_key, subkey = jax.random.split(self.rng_key)
                x_val, sig = self.update_step(x_val, jnp.array([t_i for _ in range(nr_samples)]), subkey)
                
                if self.cond_func is not None:
                    x_val = self.cond_func(x_val, sig)

            if save_evol:
                xs.append(x_val)

        if save_evol:
            return xs
        else:
            return x_val


#######################################################
#  - Sampler for the 3D data generation process       #
#  - it uses an internal conditional diffusion field  #
#  - and samples from the data distribution in an     #
#  - autoregressive way starting with 2D input image  #
#######################################################

class TemporalSampler(base):
    """
    Temporal sampler uses denoiser trained with temporal loss
    """

    def __init__(self, net_name, net_hparams, dm_name, dm_hparams, path, init_input, cond_func=None, rng_key=None):
        super().__init__(net_name, net_hparams, dm_name, dm_hparams, cond_func, rng_key)
        self.init_model(init_input, path)
        self.create_functions()

    def create_functions(self, method='ancestral', target_psnr=0.2, p_steps=2):

        self.dm.set_d_step(self.dm.N)

        if method == 'ancestral':
            def update_step(x, t, key):
                """
                Performs an update step.

                Args:
                    x (Tensor): list of input tensors (x1_t, x0_0),
                            : x1_t is the image at time t,
                            : x0_t are the images at time 0
                    t (Tensor): time tensor.

                Returns:
                    Tensor: updated tensor x1_{t-1}
                """
                if self.dm.continuous:
                    reverse = self.dm.reverse_sde
                else:
                    reverse = self.dm.reverse_discrete

                key, subkey = jax.random.split(key)
                x1_t, x0_0  = x
                if self.dm.noisy_x0:
                    x0_t        = x0_0 + self.dm.sigma_at(t)*jax.random.normal(subkey, jnp.shape(x0_0))
                else:
                    x0_t        = x0_0
                drift, diffusion= reverse(self.net.apply, self.net_weights, (x1_t, x0_t), t)
                
                # update the current image x1_t
                x[0]    = x1_t - drift + jax.random.normal(key, jnp.shape(x1_t))*diffusion
                return x, jnp.squeeze(diffusion)

        elif method == 'heun':
            def update_step(x, t, key):
                """
                Performs an update step.

                Args:
                    x (Tensor): list of input tensors (x1_t, x0_0),
                            : x1_t is the image at time t,
                            : x0_t are the images at time 0
                    t (Tensor): time tensor.

                Returns:
                    Tensor: updated tensor x1_{t-1}
                """
                if self.dm.continuous:
                    reverse = self.dm.reverse_sde
                else:
                    reverse = self.dm.reverse_discrete
                
                key, subkey = jax.random.split(key)
                x1_t, x0_0  = x
                t_next = t - 1./self.dm.N
                if self.dm.noisy_x0:
                    x0_t        = x0_0 + self.dm.sigma_at(t)*jax.random.normal(subkey, jnp.shape(x0_0))
                else:
                    x0_t        = x0_0
                drift, diffusion = reverse(self.net.apply, self.net_weights, (x1_t, x0_t), t)
                
                # update the current image x1_t
                if self.dm.noisy_x0:
                    x0_t_next = x0_0 + self.dm.sigma_at(t_next)*jax.random.normal(subkey, jnp.shape(x0_0))
                else:
                    x0_t_next = x0_0
                x1_t_first    = x1_t - drift
                drift_first, _ = reverse(self.net.apply, self.net_weights, (x1_t_first, x0_t_next), t_next)

                x[0]    = x1_t - 0.5*(drift_first + drift) + jax.random.normal(key, jnp.shape(x1_t))*diffusion
                return x, jnp.squeeze(diffusion)
            
        elif method == 'pc':

            def predictor(x, t, k):
                if self.dm.continuous:
                    reverse = self.dm.reverse_sde
                else:
                    reverse = self.dm.reverse_discrete
                drift, diffusion  = reverse(self.net.apply, self.net_weights, x, t)
                z       = jax.random.normal(k, jnp.shape(x[0]))
                x[0]    = x[0] - drift  + diffusion * z
                return x[0]

            def norm(x):
                x_squared = jnp.sum(x**2, axis=(-3, -2, -1))
                x_l2 = jnp.sqrt(x_squared)
                x_l2_keepdims = jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(x_l2, -1), -1), -1)
                return x_l2_keepdims

            def corrector(x, t, k):
                grad       = self.dm.score(self.net.apply, self.net_weights, x, t)
                noise      = jax.random.normal(k, jnp.shape(x[0]))
                grad_norm  = norm(grad)
                noise_norm = norm(noise)
                step_size  = (target_psnr*noise_norm / grad_norm) ** 2 * 2 * 1
                x[0]       = x[0] + step_size * grad + noise * jnp.sqrt(step_size * 2)
                return x[0]
            
            def update_step(x, t, key):
                
                x1_t, x0_0  = x
        
                if self.dm.noisy_x0:
                    key, subkey = jax.random.split(key)
                    x0_t        = x0_0 + self.dm.sigma_at(t)*jax.random.normal(subkey, jnp.shape(x0_0))
                else:
                    x0_t        = x0_0

                key, subkey = jax.random.split(key)
                x1_t = corrector([x1_t, x0_t], t, subkey)
                for _ in range(p_steps):
                    key, subkey = jax.random.split(key)
                    x1_t = predictor([x1_t, x0_t], t, subkey)
                x[0] = x1_t
                
                return x, None
        else:
            raise NotImplementedError

        self.update_step = jax.jit(update_step)


    def __call__(self, x_init, x0, length=2, inner_steps=1, ast_sampler=None, save_evol=False):
        """
        Perform the sampling process.

        

        """
        
        if save_evol:
            xs      = []

        nr_sequences = jnp.shape(x_init)[0]
        x = [x_init, x0]
        for _ in range(length):
            for t_i in tqdm.tqdm(jnp.linspace(self.dm.T, self.dm.eps, self.dm.N)):
                for _ in range(inner_steps):
                    self.rng_key, subkey = jax.random.split(self.rng_key)
                    x, sig = self.update_step(x, jnp.array([t_i for _ in range(nr_sequences)]), subkey)

                    if ast_sampler is not None:
                        self.rng_key, subkey = jax.random.split(self.rng_key)
                        bt = einops.rearrange(x[0], 'b t h w c -> (b t) h w c')
                        bt = ast_sampler.update_step(bt, jnp.array([t_i for _ in range(nr_sequences)]), subkey)[0]
                        x[0] = einops.rearrange(bt, '(b t) h w c -> b t h w c', b=nr_sequences)

                    if self.cond_func is not None:
                        raise NotImplementedError
                    
                    if save_evol:
                        xs.append(x[0])

        return x[0] if not save_evol else xs

def progressive_sampling(sampler, Ns, sigmaxs, rhos, x_init, inner_steps=1):

    k = len(Ns)
    x_val = x_init

    for i in range(k):

        sampler.dm.sigma_max = sigmaxs[i]
        sampler.dm.rho       = rhos[i]
        sampler.dm.set_d_step(Ns[i])
        sampler.create_functions()
        x_val = sampler(x_val, inner_steps)
    return x_val