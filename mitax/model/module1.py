from typing import Any, Callable, Optional, Tuple
from flax import linen as nn
import jax.numpy as jnp
import jax

from flax import traverse_util
from flax.linen import normalization

PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Axes = Tuple[int, ...]
Dtype = Any  

class EmbedT(nn.Module):
    """
    embedding layer using random gaussian features
    https://www.cs.cmu.edu/~schneide/DougalRandomFeatures_UAI2015.pdf
    https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
    """

    embedding_size: int = 128
    scale: float = 1.0
    param_dtype: Dtype = jnp.float32
    
    @nn.compact
    def __call__(self, t):
        
        embed_w = self.param('embed_w', jax.nn.initializers.normal(stddev=self.scale), (1, self.embedding_size,))
        embed_w = jax.lax.stop_gradient(embed_w)
        t_proj = t[:, jnp.newaxis] * embed_w * 2 * 3.1415926
        
        return jnp.concatenate([jnp.sin(t_proj), jnp.cos(t_proj)], axis=-1)

class InstanceNorm(nn.Module):
    """
    instance normalization arXiv:1907.05600
    """

    epsilon: float = 1e-6
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.ones
    reduction_axes: Axes = (1, 2)
    feature_axes: Axes = -1
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True

    @nn.compact
    def __call__(self, x):
        """Applies In normalization on the input.

        Args:
        x: the inputs

        Returns:
        Normalized inputs (the same shape as inputs).
        """
        mean, var = normalization._compute_stats(
        x,
        self.reduction_axes,
        self.dtype,
        self.axis_name,
        self.axis_index_groups,
        use_fast_variance=self.use_fast_variance)

        return normalization._normalize(
        self,
        x,
        mean,
        var,
        self.reduction_axes,
        self.feature_axes,
        self.dtype,
        self.param_dtype,
        self.epsilon,
        self.use_bias,
        self.use_scale,
        self.bias_init,
        self.scale_init,
        )

class InstanceNormPlus(nn.Module):
    """
    instance normalization arXiv:1907.05600
    """

    epsilon: float = 1e-6
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.ones
    reduction_axes: Axes = (1, 2)
    feature_axes: Axes = -1
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True

    @nn.compact
    def __call__(self, x):
        """Applies In normalization on the input.

        Args:
        x: the inputs
        
        Returns:
        Normalized inputs (the same shape as inputs).
        """
        mean, var = normalization._compute_stats(
        x, self.reduction_axes, self.dtype, self.axis_name, self.axis_index_groups, use_fast_variance=self.use_fast_variance)

        cm, cvar = normalization._compute_stats(
        mean, -1, self.dtype, self.axis_name, self.axis_index_groups, use_fast_variance=self.use_fast_variance)

        nx = normalization._normalize(self, x, mean, var, self.reduction_axes, self.feature_axes,
                                      self.dtype, self.param_dtype, self.epsilon, False, False, self.bias_init, self.scale_init)

        n_mean = normalization._normalize(self, mean, cm, cvar, -1, self.feature_axes, 
                                                 self.dtype, self.param_dtype, self.epsilon, False, False, self.bias_init, self.scale_init)
        
        n_mean_scale = self.param('n_mean_scale', self.scale_init, jnp.shape(x)[-1], self.param_dtype)
        mix_x = n_mean_scale*(jnp.expand_dims(n_mean, self.reduction_axes)) + nx
        
        return normalization._normalize(
        self,
        mix_x,
        mean,
        var,
        self.reduction_axes,
        self.feature_axes, self.dtype, self.param_dtype, self.epsilon, self.use_bias, self.use_scale, self.bias_init, self.scale_init)  

class InstanceNorm2dPlus(nn.Module):
  """InstanceNorm++ as proposed in the original NCSN paper."""
  bias: bool = True

  @staticmethod
  def scale_init(key, shape, dtype=jnp.float32):
    normal_init = nn.initializers.normal(0.02)
    return normal_init(key, shape, dtype=dtype) + 1.

  @nn.compact
  def __call__(self, x):
    means = jnp.mean(x, axis=(1, 2))
    m = jnp.mean(means, axis=-1, keepdims=True)
    v = jnp.var(means, axis=-1, keepdims=True)
    means_plus = (means - m) / jnp.sqrt(v + 1e-5)

    h = (x - means[:, None, None, :]) / jnp.sqrt(jnp.var(x, axis=(1, 2), keepdims=True) + 1e-5)

    h = h + means_plus[:, None, None, :] * self.param('alpha', InstanceNorm2dPlus.scale_init, (1, 1, 1, x.shape[-1]))
    h = h * self.param('gamma', InstanceNorm2dPlus.scale_init, (1, 1, 1, x.shape[-1]))
    if self.bias:
      h = h + self.param('beta', nn.initializers.zeros, (1, 1, 1, x.shape[-1]))

    return h

class CrpBlock(nn.Module):
    """
    chained residual pool block
    """
    nr_filters: Optional[int] = None
    nr_stages: int = 2
    nonlinearity: Callable = nn.relu
    normalizer: Callable = InstanceNorm

    @nn.compact
    def __call__(self, x):
        x = self.nonlinearity(x)
        path = x

        if self.nr_filters is None:
            nr_filters = jnp.shape(x)[-1]
        else:
            nr_filters = self.nr_filters

        for _ in range(self.nr_stages):
            path = self.normalizer()(path)
            path = nn.avg_pool(path, window_shape=(5,5), strides=(1,1), padding="SAME") #avg_pool2d
            path = nn.Conv(features=nr_filters, kernel_size=(3,3), use_bias=False)(path)  # don't need bias 
            x = path + x
        return x

class CondRcuBlock(nn.Module):
    """
    residual convolution unit
    """
    nr_filters: Optional[int] = None
    nr_resnet: int = 2
    nr_stages: int = 2
    nonlinearity: Callable = nn.relu
    normalizer: Callable = InstanceNorm()

    @nn.compact
    def __call__(self, x, h):
        if self.nr_filters is None:
            nr_filters = jnp.shape(x)[-1]
        else:
            nr_filters = self.nr_filters
        for _ in range(self.nr_resnet):
            residual = x
            for _ in range(self.nr_stages):
                x = self.normalizer()(x)
                x = self.nonlinearity(x)
                x = nn.Conv(features=nr_filters, kernel_size=(3, 3), use_bias=False)(x)  # don't need bias
            x += residual
        if h is not None:
            h = nn.WeightNorm(nn.Dense(features=nr_filters))(h)
            h = self.nonlinearity(h)
            x = x + jnp.expand_dims(h, (1, 2))
        return x
    

class MsfBlock(nn.Module):
    """
    multi-resolution fusion
    """
    nr_filters: Optional[int] = None
    out_shape: Tuple[int, ...] = (1, 1)
    resize_method: str = 'nearest'
    normalizer: Callable = InstanceNorm()

    @nn.compact
    def __call__(self, blocks):
        sums = []
        for i in range(len(blocks)):
            xl_out = self.normalizer()(blocks[i])
            if self.nr_filters is None:
                self.nr_filters = jnp.shape(blocks[i])[-1]
            xl_out = nn.Conv(features=self.nr_filters, kernel_size=(3, 3))(xl_out)
            xl_out = jax.image.resize(xl_out, list(self.out_shape)+[self.nr_filters], method=self.resize_method)
            sums.append(xl_out)

        return jax.tree_util.tree_map(lambda *args: sum(args), *sums) #TODO: maybe there is a better way to do this 
        #return sum(sums)

class RefineBlock(nn.Module):
    """
    refine block
    """

    nr_filters: Optional[int] = None
    out_shape: Tuple[int, ...] = (32, 32)
    resize_method: str = 'nearest'
    normalizer: Callable = InstanceNorm
    nonlinearity: Callable = nn.elu
    nr_stage: int = 2

    @nn.compact
    def __call__(self, blocks, h, end=False):
        outs = []

        for i in range(len(blocks)):
            outs.append(CondRcuBlock(nr_filters=None, nr_resnet=2, nr_stages=self.nr_stage, nonlinearity=self.nonlinearity, normalizer=self.normalizer)(blocks[i], h))
        
        if len(blocks) > 1:
            y = MsfBlock(nr_filters=self.nr_filters, out_shape=self.out_shape, normalizer=self.normalizer, resize_method=self.resize_method)(outs)
        else:
            y = outs[0]
        
        y = CrpBlock(nr_filters=None, nr_stages=self.nr_stage, nonlinearity=self.nonlinearity, normalizer=self.normalizer)(y)

        if end:
            y = CondRcuBlock(nr_filters=self.nr_filters, nr_resnet=3, nr_stages=self.nr_stage, nonlinearity=self.nonlinearity, normalizer=self.normalizer)(y, h)
        else:
            y = CondRcuBlock(nr_filters=self.nr_filters, nr_resnet=1, nr_stages=self.nr_stage, nonlinearity=self.nonlinearity, normalizer=self.normalizer)(y, h)
        

        return y

class ResBlock(nn.Module):
    
    """
    resnet block
    out_filters is output_dims/feature
    """

    out_filters: int
    nonlinearity: Callable = nn.elu
    normalizer: Callable = InstanceNorm
    rescale: bool = False
    dropout: float = 0.
    dilation: bool = False


    @nn.compact
    def __call__(self, x, h):
        in_filters = jnp.shape(x)[-1]
        x_skip = x
        x = self.normalizer()(x) 
        x = self.nonlinearity(x)
        if self.rescale:
            x = nn.Conv(features=in_filters, kernel_size=(3,3))(x)
        else:
            x = nn.Conv(features=self.out_filters, kernel_size=(3,3))(x)

        x = self.normalizer()(x)
        x = self.nonlinearity(x)

        if self.dropout > 0.0:
            x = nn.Dropout(rate=self.dropout)(x)

        x = nn.Conv(self.out_filters, kernel_size=(3,3))(x) # nonlinearity=None
        if not self.dilation and self.rescale:
            x = nn.avg_pool(x, window_shape=(2,2), strides=(2,2), padding='SAME')
        
        if self.out_filters == in_filters and not self.rescale:
            shortcut = x_skip
        else:
            if not self.dilation:
                shortcut = nn.Conv(features=self.out_filters, kernel_size=[1,1])(x_skip) #  nonlinearity=None
                shortcut = nn.avg_pool(shortcut, window_shape=(2,2), strides=(2,2), padding='SAME')
            else:
                shortcut = nn.Conv(features=self.out_filters, kernel_size=(3,3))(x_skip) # without nonlinearity

        if h is not None:
            h = nn.WeightNorm(nn.Dense(features=self.out_filters))(h)
            h = self.nonlinearity(h)
            if self.dropout>0.:
                h = nn.Dropout(rate=self.dropout)(h)
            shortcut = shortcut + jnp.expand_dims(h, (1, 2))

        return shortcut + x

if __name__ == '__main__':
    
    def test_EmbedT():

        t = jnp.array([0.1, 0.2, 0.3])  # Example input
        o_embed = EmbedT(embedding_size=32)
        variables = o_embed.init(jax.random.PRNGKey(0), t)
        result = o_embed.apply(variables, t)

        assert result.shape == (3, 2 * o_embed.embedding_size)  # Check the shape of the output
    
    def test_InstanceNormPlus():
        x = jnp.ones((1, 2, 2, 1))
        o_norm = InstanceNormPlus()
        variables = o_norm.init(jax.random.PRNGKey(0), x)
        result = o_norm.apply(variables, x)
        assert result.shape == x.shape

    test_EmbedT()
    test_InstanceNormPlus()