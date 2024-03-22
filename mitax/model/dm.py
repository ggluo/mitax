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

    def __init__(self, sigma_max, sigma_min, sigma_type='power', N=1000, T=1., eps=1.e-5, rho=7, continuous=True, weighting=False, reduce_mean=False):
        super().__init__(sigma_max, sigma_min, sigma_type, N, T, eps, rho, continuous) # continuous work well for SMLD sde
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
            sigma = self.sigma_min + (self.sigma_max - self.sigma_min)*jnp.log((jnp.exp(1.)-1.)*t + 1.)
        elif self.sigma_type == 'sqrt':
            sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * t**0.5
        elif self.sigma_type == 'rho':
            sigma = (self.sigma_min**(1/self.rho)+ t*(self.sigma_max**(1/self.rho) - self.sigma_min**(1/self.rho)))**self.rho
        else:
            raise TypeError("Check you the type of sigma_t!")
        return sigma

    def sigma_t(self, t):
        """
        noise schedule for t in (1, 0), distribution of sigma 
        """
        if not isinstance(t, jnp.ndarray) :
            if isinstance(t, list):
                t = jnp.array(t)
            else:
                t = jnp.array([t])
        else:
            if t.shape == ():
                t = jnp.array([t])
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
    the reparamatrization trick in arXiv 2206.00364 (Tero Karras)
    """
    def __init__(self, sigma_max, sigma_min, sigma_type='rho', N=1000, T=1., eps=1.e-5, rho=7, continuous=True, weighting=True, reduce_mean=False):
        super().__init__(sigma_max, sigma_min, sigma_type, N, T, eps, rho, continuous, weighting, reduce_mean)
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

    def map_sigma(self, s):
        return jnp.exp((s+self.loc)*self.scale)  # not tested

    def __call__(self, apply_fn, params, inputs, training):
        """
        loss function to training the denoiser
        inputs is a tuple or list that contains the clean image from a dataset and a random key
        """
        x, key = inputs

        z         = jax.random.normal(key, jnp.shape(x))
        sigma     = self.map_sigma(jax.random.normal(key, [x.shape[0]])) # haven't test this map_sigma function

        if training:
            dropout_key, key = jax.random.split(key)

        denoise_x  = self.denoiser(apply_fn, params, x + batch_mul(z, sigma), sigma, training, dropout_key if training else None)

        reduce = lambda tmp: jnp.mean(tmp, axis=[1,2,3]) if self.reduce_mean else jnp.sum(tmp, axis=[1,2,3])

        l2 = jnp.square(denoise_x - x)
        
        l = jnp.mean(reduce(l2) if not self.weighting else reduce(l2)*self.get_weights(sigma))

        return l, {}


class temporal_x(smld):
    """
    """
    def __init__(self, sigma_max, sigma_min, sigma_type='rho', N=1000, T=1., eps=1.e-5, rho=7, continuous=True, weight_order=0, test_scalings=False, map_sigmas=False, reduce_mean=False, noisy_x0=True):
        super().__init__(sigma_max, sigma_min, sigma_type, N, T, eps, rho, continuous, True, reduce_mean)
        self.weight_order = weight_order
        self.map_sigmas = map_sigmas
        self.test_scalings = test_scalings
        self.sigma_data = 0.5  # TODO: compute sigma_data with the given dataset
        self.loc = -1.5
        self.scale = 1.5
        self.noisy_x0 = noisy_x0

    def get_weights(self, sigma, weight_order):
        if weight_order == 1:
            return 1./sigma + 4.
        if weight_order == 2:
            return 1./sigma**2 + 4.
        if weight_order == 0:
            return 1.

    def get_scalings(self, sigma):
        # used to mix x0_t and x1_t
        # x0_t and x1_t are the noisy images at frame(0,1)
        # x0_0 and x1_0 are the clean images at frame(0,1)
        # the learned map is map(c_out * F_\theta(x0_t*c_in) + c_skip * x1_t) with reparametrization trick
        # when t is 1, c_skip is 0, c_out is 1, where sigma is the largest
        # it is easy for the learned model to map x0_t to x1_0, and this mapping process is to get structure information of x1_0
        # as x0_t has less structure information of x0_0 and temporal prior information in the learned mapping can dominate this mapping process
        # (using map maybe not a good idea, how about move or nudge?)
        # when t is 0, c_skip is 1, c_out is 0, where sigma is the smallest 
        # it is difficult for the learned model to map x0_0 to x_1_0, that is to say, it is difficult to get the structure information of x1_0
        # as x0_0 has all the structure information and temporal prior information in the learned mapping can't dominate this mapping process
        # and we already incrementally got structure information of x1_0 from the mapping process when t > 0 
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out  = sigma * self.sigma_data / jnp.sqrt(sigma**2 + self.sigma_data**2)
        c_in   = 1 / jnp.sqrt(sigma**2 + self.sigma_data**2) # a function of sigma is needed to destory the structure information of x0_t
        return c_skip[:, jnp.newaxis, jnp.newaxis, jnp.newaxis], c_out[:, jnp.newaxis, jnp.newaxis, jnp.newaxis], c_in[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
    
    def map_sigma(self, s):
        return jnp.exp((s+self.loc)*self.scale)

    def sde(self, x, t):
        return super().sde(x[0], t)

    def score(self, apply_fn, weights, xs, t):
        # xs is a tuple or list that contains x1_t and x0_t
        # x0_t is the noisy image from previous frames
        # return the score of x1_t given x0_t
        x1_t, x0_t = xs
        sigma = self.sigma_at(t)
        if self.test_scalings:
            dif = self.denoiser_2(apply_fn, weights, x0_t, x1_t, sigma, train=False, key=None)-x1_t
        else:
            dif = self.denoiser(apply_fn, weights, x0_t, x1_t, sigma, train=False, key=None)-x1_t
        return dif/sigma[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]**2

    def denoiser(self, apply_fn, weights, x0_t, x1_t, sigma, train, key):
        c_skip, c_out, c_in = self.get_scalings(sigma)
        x_comb = x0_t * c_out + x1_t * c_skip
        model_output = apply_fn({'params': weights}, **{'x': c_in*x_comb, 't': sigma}, train=train, rngs={'dropout': key} if train else None)
        return c_out * model_output + c_skip * x1_t


    def denoiser_2(self, apply_fn, weights, x0_t, x1_t, sigma, train, key):
        c_skip, c_out, c_in = self.get_scalings(sigma)
        x_comb = x0_t * c_out + x1_t * (1.-c_out)
        model_output = apply_fn({'params': weights}, **{'x': c_in*x_comb, 't': sigma}, train=train, rngs={'dropout': key} if train else None)
        return c_out * model_output + c_skip * x1_t
    
    def __call__(self, apply_fn, params, inputs, training):
        """
        loss function for temporal diffusion field
        x0 is the clean image from a dataset
        t is the sigma used to perturb x_0
        x1 is the clean image that x_0 is mapped to
        """
        x0, x1, key = inputs

        x_shape    = jnp.shape(x0)
        key, subkey= jax.random.split(key, 2)
        z1         = jax.random.normal(subkey, x_shape)

        # if map sigma is false, the range of sigma is [sigma_min, sigma_max]
        # else the range of sigma is determined by map_sigma(loc, scale)
        if not self.map_sigmas:
            key, subkey = jax.random.split(key)
            t     = jax.random.uniform(subkey, (x_shape[0],), minval=self.eps, maxval=self.T)
            sigma = self.sigma_at(t)
        else:
            sigma = self.map_sigma(jax.random.normal(key, [x_shape[0]]))

        std = sigma[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]

        x1_t = x1 + std * z1
        if self.noisy_x0:
            key , z_key = jax.random.split(key)
            z           = jax.random.normal(z_key, x_shape)
            x0_t = x0 + std * z
        else:
            x0_t = x0

        if training:
            dropout_key, key = jax.random.split(key)

        if self.test_scalings:
            pred_x1     = self.denoiser_2(apply_fn, params, x0_t, x1_t,  sigma, train=training, key=dropout_key if training else None)
        else:
            pred_x1     = self.denoiser(apply_fn, params, x0_t, x1_t,  sigma, train=training, key=dropout_key if training else None)

        l2 = jnp.square(pred_x1 - x1) * self.get_weights(sigma, self.weight_order)

        reduce    = lambda tmp: jnp.mean(tmp, axis=[1,2,3]) if self.reduce_mean else jnp.sum(tmp, axis=[1,2,3])
        l = jnp.mean(reduce(l2))

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
    def test_loss(x, key):
        
        net = indentity()
        params = net.init(key, x, None, train=False)
        
        return power_smld.sigma_at(t), power_smld(net, params, (x, key), True)

    
    key = jax.random.PRNGKey(100)
    ls = []
    s = 10
    for i in range(s):
        key, sey = jax.random.split(key)
        sigma, l = test_loss(jnp.zeros((1, 2, 2, 1)), sey)
        ls.append(l[0])
    print(sigma)
    print(jnp.mean(jnp.array(ls)), jnp.var(jnp.array(ls)))
