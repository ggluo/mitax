from flax import linen as nn
import jax.numpy as jnp
import jax
class mnist_cnn(nn.Module):
    """A simple cnn model for mnist classifying"""
    dropout: float = 0.1
    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Conv(features=32, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2,2), strides=(2,2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout)(x, deterministic=not train)
        x = nn.Dense(features=10)(x)
        return x


class dm_cnn(nn.Module):
    """A simple cnn model for diffusion model"""

    @nn.compact
    def __call__(self, x, t):
        in_size = x.shape[1]
        n_hidden = 256
        
        t = jnp.concatenate([t - 0.5, jnp.cos(2*jnp.pi*t), jnp.sin(2*jnp.pi*t), -jnp.cos(4*jnp.pi*t)],axis=1)
        x = jnp.concatenate([x, t],axis=1)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(in_size)(x)
        x = jax.vmap(x/t)
        return x

class Gaussian_Mixture():
    
    def __init__ (self, dim, mix_prob=[0.7, 0.2, 0.1]):

        self.mix_prob = mix_prob
        self.means    = jnp.stack([5*jnp.ones(dim), -5*jnp.ones(dim), [5,-5]], axis=0)
        self.sigma    = 1.

    def sample(self, nr_samples, key, sigma=1.):
        key, subkey = jax.random.split(key)
        mix_idx = jax.random.categorical(key, jnp.log([self.mix_prob]), nr_samples)
        means   = jnp.take(self.means, mix_idx)
        return jax.random.normal(subkey, means.shape)*sigma + means

    def log_prob(self, samples, sigma=1.):
        logps = []
        for i in range(len(self.mix_prob)):
            logps.append((-jnp.sum((samples - self.means[i]) ** 2, axis=-1) / (2 * sigma ** 2) - 0.5 * jnp.log(
                2 * jnp.pi * sigma ** 2)) + jnp.log(self.mix_prob[i]))
        logp = jax.scipy.special.logsumexp(jnp.stack(logps, axis=0), axis=0)
        return logp

    def score(self, samples, sigma=1.):
        log_probs = jnp.sum(self.log_prob(samples, sigma))
        return jax.vmap(jax.grad(log_probs, samples))