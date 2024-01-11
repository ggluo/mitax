import sys
sys.path.append('/home/gluo/mitax')

from typing import Any
import jax.numpy as jnp
import jax
from functools import partial
from flax import linen as nn
from mitax.misc.utils import batch_mul

class base():

    def __init__(self, sigma_max, sigma_min, sigma_type='exp', N=1000, T=1., eps=1.e-5, rho=7, continuous=True): 

        self.sigma_min  = sigma_min
        self.sigma_max  = sigma_max
        self.sigma_type = sigma_type
        self.N          = N
        self.T          = T
        self.eps        = eps
        self.continuous = continuous
        self.rho        = rho
        self.sigmas     = self.cal_sigmas()
    def __call__(self, *args: Any, **kwds: Any) -> Any:
            raise NotImplementedError

class smld(base):
    """
    Variance exploded diffusion model
    Ref: ICLR2021 (Song Yang) --> SMLD
    """

    def __init__(self, sigma_max, sigma_min, sigma_type='power', N=1000, T=1., eps=1.e-5, continuous=True, weighting=False, reduce_mean=False):
        super().__init__(sigma_max, sigma_min, sigma_type, N, T, eps, continuous) # continuous work well for SMLD sde
        self.weighting   = weighting
        self.reduce_mean = reduce_mean

    def sigma_func(self, t):
        if self.sigma_type == "quad":
            sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * t**2
        elif self.sigma_type == "power":
            sigma = self.sigma_min * (self.sigma_max / self.sigma_min)**t
        elif self.sigma_type == 'linear':
            sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * t
        elif self.sigma_type == 'log':
            sigma = self.sigma_min + (self.sigma_max - self.sigma_min)*jnp.math.log((jnp.math.exp(1.)-1.)*t + 1.)
        elif self.sigma_type == 'sqrt':
            sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * t**0.5
        elif self.sigma_type == 'rho_p':
            sigma = (self.sigma_min**(1/self.rho)+ t*(self.sigma_max**(1/self.rho) - self.sigma_min**(1/self.rho)))**self.rho
        else:
            raise TypeError("Check you the type of sigma_t!")
        return sigma

    def sigma_t(self, t):
        """
        noise schedule for t in (1, 0), distribution of sigma 
        """
        return jax.vmap(self.sigma_func)(t)
    
    #@partial(jax.jit, static_argnums=(0,))
    def sigmas_t_for_grad(self, t):
        """
        noise schedule for t in (1, 0), distribution of sigma 
        """
        return jnp.square(self.sigma_func(t))

    def cal_sigmas(self):
        # discrete sigma
        return self.sigma_t(jnp.linspace(self.T, self.eps, self.N+1))

    def set_d_step(self, N):
        # to update sigmas after changing N and sigma_max, rho
        self.N = N
        self.sigmas = self.cal_sigmas()

    def sigma_at(self, t):
        if self.continuous:
            sigma = self.sigma_t(t)
        else:
            sigma = jnp.take(self.sigmas, jnp.array(list(map(jnp.int32, ((1. - t / self.T) * self.N)))))
        return sigma
    
    def sde(self, x, t):
        #https://stackoverflow.com/questions/68609584/jax-problem-in-differentiating-of-function
        drift     = jnp.zeros_like(x)
        grad      = jax.vmap(jax.grad((self.sigmas_t_for_grad))) # TODO: check the grad
        diffusion = jnp.sqrt(grad(t)/self.N)

        return drift, diffusion[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]

    def reverse_sde(self, apply_fn, weights, x, t, ode=False):
        
        drift, diffusion = self.sde(x, t)
        if ode:
            score = 0.5*self.score(apply_fn, weights, x, t)
        else:
            score = self.score(apply_fn, weights, x, t)
        drift = drift - diffusion ** 2 * score

        return drift, diffusion

    def discretize(self, x, t):

        timestep  = jnp.array(list(map(jnp.int32, ((1. - t / self.T) * self.N))))
        sigma     = jnp.take(self.sigmas, timestep)
        adj_sigma = jnp.take(self.sigmas, timestep+1)

        f         = jnp.zeros_like(x)
        g         = jnp.sqrt(sigma ** 2 - adj_sigma ** 2)

        return f, g[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]

    def reverse_discrete(self, apply_fn, weights, x, t):
        f, G  = self.discretize(x, t)
        rev_f = f - G ** 2 * self.score(apply_fn, weights, x, t)
        rev_g = G 
        return rev_f, rev_g


    def score(self, apply_fn, weights, x_t, t):
        sigma = self.sigma_at(t)
        return apply_fn({'params': weights}, **{'x': x_t, 't': sigma}, train=False)

    def __call__(self, apply_fn, params, inputs, training):
        """
        loss function for SMLD
        x is the clean image from a dataset
        t is the sigma used to perturb image
        """
        x, key = inputs

        x_shape    = jnp.shape(x)
        t          = jax.random.uniform(key, (x_shape[0],), minval=self.eps, maxval=self.T)
        z          = jax.random.normal(key, x_shape)

        sigma     = self.sigma_at(t)
        std       = sigma[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]

        if training:
            dropout_key, key = jax.random.split(key)

        score     = apply_fn({'params': params}, x=x + std * z, t=sigma, train=training,
                             rngs={'dropout': dropout_key} if training else None)

        if not self.weighting:
            l2 = jnp.square(score * std + z)
        else:
            l2 = jnp.square(score + z / std)

        reduce    = lambda tmp: jnp.mean(tmp, axis=[1,2,3]) if self.reduce_mean else jnp.sum(tmp, axis=[1,2,3])
        l = jnp.mean(reduce(l2))

        return l, {}

class smld_x(smld):
    """
    elucidating the design space of diffusion-based generative models
    arXiv 2206.00364 (Tero Karras)
    """
    def __init__(self, sigma_max, sigma_min, sigma_type='rho_p', N=1000, T=1., eps=1.e-5, rho=7, continuous=True, weighting=True, reduce_mean=False):
        super().__init__(sigma_max, sigma_min, sigma_type, N, T, eps, continuous, weighting, reduce_mean)
        self.rho = rho
        self.sigma_data = 0.5  # TODO: compute sigma_data with the given dataset
        self.loc = -1.5
        self.scale = 1.5

    def score(self, apply_fn, weights, x_t, t):
        sigma = self.sigma_at(t)
        dif = self.denoiser(apply_fn, weights, x_t, sigma, train=False)-x_t
        return dif/sigma[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]**2
    
    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out  = sigma * self.sigma_data / jnp.sqrt(sigma**2 + self.sigma_data**2)
        c_in   = 1 / jnp.sqrt(sigma**2 + self.sigma_data**2)
        return c_skip[:, jnp.newaxis, jnp.newaxis, jnp.newaxis], c_out[:, jnp.newaxis, jnp.newaxis, jnp.newaxis], c_in[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]

    def denoiser(self, apply_fn, weights, x_t, sigma, train, key=None):
        c_skip, c_out, c_in = self.get_scalings(sigma)
        model_output = apply_fn({'params': weights}, **{'x': c_in*x_t, 't': sigma}, train=train, rngs={'dropout': key} if key is not None else None)
        return c_out * model_output + c_skip * x_t

    def get_weights(self, sigma, weight_order=2):
        if weight_order == 1:
            return 1./sigma + 4.
        if weight_order == 2:
            return 1./sigma**2 + 4.

    def __call__(self, apply_fn, params, inputs, training):
        """
        loss function to training the denoiser
        inputs is a tuple or list that contains the clean image from a dataset and a random key
        """
        x, key = inputs

        z         = jax.random.normal(key, jnp.shape(x))
        s         = jax.random.normal(key, [x.shape[0]])
        sigma     = jnp.exp((s+self.loc)*self.scale)

        if training:
            dropout_key, key = jax.random.split(key)

        denoise_x  = self.denoiser(apply_fn, params, x + batch_mul(z, sigma), sigma, training, dropout_key if training else None)

        reduce = lambda tmp: jnp.mean(tmp, axis=[1,2,3]) if self.reduce_mean else jnp.sum(tmp, axis=[1,2,3])

        l2 = jnp.square(denoise_x - x)
        
        l = jnp.mean(reduce(l2) if not self.weighting else reduce(l2)*self.get_weights(sigma))

        return l, {}


if __name__ == '__main__':
    power_smld = smld(sigma_max=3., sigma_min=0.001, sigma_type='power', N=50, T=1., eps=1.e-5, continuous=True, weighting=False, reduce_mean=False)

    t = jnp.linspace(power_smld.T, power_smld.eps, power_smld.N)
    import matplotlib.pyplot as plt
    plt.plot(t, power_smld.sigma_t(t))
    plt.show()
    a = power_smld.sde(jnp.ones_like(t), t)
    
    b = power_smld.discretize(jnp.ones_like(t), t)
    
    print(power_smld.sigma_at(jnp.array([0.0,0.5, 1.0])))

    class indentity(nn.Module):
        """A simple cnn model for mnist classifying"""

        @nn.compact
        def __call__(self, x, t, train=True):
            return x
    
    @jax.jit
    def test_loss(x, t, key):
        
        net = indentity()
        params = net.init(key, x, None, train=False)
        
        return power_smld.sigma_at(t), power_smld(net, params, (x, t, key), True)

    
    key = jax.random.PRNGKey(100)
    ls = []
    s = 10
    for i in range(s):
        key, sey = jax.random.split(key)
        sigma, l = test_loss(jnp.zeros((1, 2, 2, 1)), jnp.array([0.]), sey)
        ls.append(l[0])
    print(sigma)
    print(jnp.mean(jnp.array(ls)), jnp.var(jnp.array(ls)))