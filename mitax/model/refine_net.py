from flax import linen as nn
import jax.numpy as jnp
import jax

import sys
sys.path.append('/home/gluo/mitax')

from mitax.model.module1 import RefineBlock, ResBlock, InstanceNormPlus, EmbedT, InstanceNorm2dPlus
from typing import Callable
class base(nn.Module):
    """
    Initialize the RefineNet model.

    Args:
        chns (int): Number of input channels.
        nr_filters (int): Number of filters in the convolutional layers.
        nonlinearity (str): Nonlinearity function to use.
        fourier_scale (float): Fourier scale for the attention mechanism.
        affine_x (bool): Whether to apply affine transformation to the input.
        attention (bool): Whether to use attention mechanism.
        dropout (float, optional): Dropout rate. Defaults to 0.
        normalizer (str, optional): Normalization method. Defaults to 'InstanceNormPlus'.
        scale_out (bool, optional): Whether to scale the output. Defaults to True.
    """

    chns: int # remove this argument
    nr_filters: int
    affine_x: bool = False
    fourier_scale: int = 32
    attention: bool = False
    dropout: float = 0.
    scale_out: bool = True
    normalizer: Callable = InstanceNorm2dPlus
    nonlinearity: Callable = nn.elu

    @nn.compact
    def __call__(self, x, t):
        raise NotImplementedError

class tiny(base):

    @nn.compact
    def __call__(self, x, t, train=True):

        if self.affine_x:
            x = 2*x - 1
        
        proj_t = EmbedT(embedding_size=self.nr_filters, scale=self.fourier_scale)(jnp.log(t))

        proj_t = nn.Dense(features=self.nr_filters*2)(proj_t)
        proj_t = self.nonlinearity(proj_t)
        proj_t = nn.Dense(features=self.nr_filters*4)(proj_t)

        x_level_0   = nn.Conv(features=1*self.nr_filters, kernel_size=(3,3))(x)
        x_level_1_0 = ResBlock(out_filters=1*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, dropout=self.dropout, rescale=False)(x_level_0, h=proj_t)
        x_level_1_1 = ResBlock(out_filters=1*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, dropout=self.dropout, rescale=False)(x_level_1_0, h=proj_t)
        x_level_2_0 = ResBlock(out_filters=2*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, dropout=self.dropout, rescale=True)(x_level_1_1, h=proj_t)
        x_level_2_1 = ResBlock(out_filters=2*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, dropout=self.dropout, rescale=False)(x_level_2_0, h=proj_t)
        x_level_3_0 = ResBlock(out_filters=2*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, dropout=self.dropout, rescale=True, dilation=2)(x_level_2_1, h=proj_t)
        x_level_3_1 = ResBlock(out_filters=2*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, dropout=self.dropout, rescale=False, dilation=2)(x_level_3_0, h=proj_t)
        
        refine_1 = RefineBlock(nr_filters=2*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, out_shape=jnp.shape(x_level_3_1)[0:3])([x_level_3_1], h=proj_t)
        refine_2 = RefineBlock(nr_filters=1*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, out_shape=jnp.shape(x_level_2_1)[0:3])([x_level_2_1, refine_1], h=proj_t)
        refine_3 = RefineBlock(nr_filters=1*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, out_shape=jnp.shape(x_level_1_1)[0:3])([x_level_1_1, refine_2], h=proj_t, end=True)
        

        out = self.normalizer()(refine_3)
        out = self.nonlinearity(out)
        out = nn.Conv(features=self.chns, kernel_size=(3,3))(out)
        
        if self.scale_out:
            out = out / t[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
        return out

class small(base):

    
    @nn.compact
    def __call__(self, x, t, train=True):
        """
        multi level refine net conditional on sigma_t, t
        """
        if self.affine_x:
            x = 2*x - 1
            
        proj_t = EmbedT(embedding_size=self.nr_filters, scale=self.fourier_scale)(jnp.log(t))

        proj_t = nn.WeightNorm(nn.Dense(features=self.nr_filters * 2))(proj_t)
        proj_t = self.nonlinearity(proj_t)
        proj_t = nn.WeightNorm(nn.Dense(features=self.nr_filters*4))(proj_t)

        x_level_0   = nn.Conv(features=1*self.nr_filters, kernel_size=(3,3))(x)
        x_level_1_0 = ResBlock(out_filters=1*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, dropout=self.dropout, rescale=False)(x_level_0, h=proj_t)
        x_level_1_1 = ResBlock(out_filters=1*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, dropout=self.dropout, rescale=False)(x_level_1_0, h=proj_t)
        x_level_2_0 = ResBlock(out_filters=2*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, dropout=self.dropout, rescale=True)(x_level_1_1, h=proj_t)
        x_level_2_1 = ResBlock(out_filters=2*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, dropout=self.dropout, rescale=False)(x_level_2_0, h=proj_t)
        x_level_3_0 = ResBlock(out_filters=2*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, dropout=self.dropout, rescale=True, dilation=2)(x_level_2_1, h=proj_t)
        x_level_3_1 = ResBlock(out_filters=2*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, dropout=self.dropout, rescale=False, dilation=2)(x_level_3_0, h=proj_t)
        x_level_4_0 = ResBlock(out_filters=2*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, dropout=self.dropout, rescale=True, dilation=4)(x_level_3_1, h=proj_t)
        x_level_4_1 = ResBlock(out_filters=2*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, dropout=self.dropout, rescale=False, dilation=4)(x_level_4_0, h=proj_t)

        refine_0 = RefineBlock(nr_filters=2*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, out_shape=jnp.shape(x_level_4_1)[0:3])([x_level_4_1], h=proj_t)
        refine_1 = RefineBlock(nr_filters=2*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, out_shape=jnp.shape(x_level_3_1)[0:3])([x_level_3_1, refine_0], h=proj_t)
        refine_2 = RefineBlock(nr_filters=1*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, out_shape=jnp.shape(x_level_2_1)[0:3])([x_level_2_1, refine_1], h=proj_t)
        refine_3 = RefineBlock(nr_filters=1*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, out_shape=jnp.shape(x_level_1_1)[0:3])([x_level_1_1, refine_2], h=proj_t, end=True)
        

        out = self.normalizer()(refine_3)
        out = self.nonlinearity(out)
        out = nn.Conv(features=self.chns, kernel_size=(3,3))(out)
        
        if self.scale_out:
            out = out / t[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
        return out
    
class big(base):
    
    @nn.compact
    def __call__(self, x, t, train=True):
        
        if self.affine_x:
            x = 2*x - 1
        
        proj_t = EmbedT(embedding_size=self.nr_filters, scale=self.fourier_scale)(jnp.log(t))

        proj_t = nn.WeightNorm(nn.Dense(features=self.nr_filters * 2))(proj_t)
        proj_t = self.nonlinearity(proj_t)
        proj_t = nn.WeightNorm(nn.Dense(features=self.nr_filters*4))(proj_t)

        x_level_0   = nn.Conv(features=1*self.nr_filters, kernel_size=(3,3))(x)
        x_level_1_0 = ResBlock(out_filters=1*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, dropout=self.dropout, rescale=False)(x_level_0, h=proj_t)
        x_level_1_1 = ResBlock(out_filters=1*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, dropout=self.dropout, rescale=False)(x_level_1_0, h=proj_t)
        x_level_2_0 = ResBlock(out_filters=2*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, dropout=self.dropout, rescale=True)(x_level_1_1, h=proj_t)
        x_level_2_1 = ResBlock(out_filters=2*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, dropout=self.dropout, rescale=False)(x_level_2_0, h=proj_t)
        x_level_3_0 = ResBlock(out_filters=2*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, dropout=self.dropout, rescale=False, dilation=2)(x_level_2_1, h=proj_t)
        x_level_3_1 = ResBlock(out_filters=2*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, dropout=self.dropout, rescale=True)(x_level_3_0, h=proj_t)
        if self.attention:
            x_level_3_1 = nn.self_attention(x_level_3_1, qk_chns=2*self.nr_filters, v_chns=2*self.nr_filters)
        x_level_4_0 = ResBlock(out_filters=2*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, dropout=self.dropout, rescale=False, dilation=4)(x_level_3_1, h=proj_t)
        x_level_4_1 = ResBlock(out_filters=2*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, dropout=self.dropout, rescale=True)(x_level_4_0, h=proj_t)
        if self.attention:
            x_level_4_1 = nn.self_attention(x_level_4_1, qk_chns=2*self.nr_filters, v_chns=2*self.nr_filters)
        x_level_5_0 = ResBlock(out_filters=2*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, dropout=self.dropout, rescale=False, dilation=4)(x_level_4_1, h=proj_t)
        x_level_5_1 = ResBlock(out_filters=2*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, dropout=self.dropout, rescale=True)(x_level_5_0, h=proj_t)
        if self.attention:
            x_level_5_1 = nn.self_attention(x_level_5_1, qk_chns=2*self.nr_filters, v_chns=2*self.nr_filters)

        refine_0 = RefineBlock(nr_filters=2*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, out_shape=jnp.shape(x_level_5_1)[0:3])([x_level_5_1], h=proj_t)
        if self.attention:
            refine_0 = nn.self_attention(refine_0, qk_chns=2*self.nr_filters, v_chns=2*self.nr_filters)
        refine_1 = RefineBlock(nr_filters=2*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, out_shape=jnp.shape(x_level_4_1)[0:3])([x_level_4_1, refine_0], h=proj_t)
        if self.attention:
            refine_1 = nn.self_attention(refine_1, qk_chns=2*self.nr_filters, v_chns=2*self.nr_filters)
        refine_2 = RefineBlock(nr_filters=2*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, out_shape=jnp.shape(x_level_3_1)[0:3])([x_level_3_1, refine_1], h=proj_t)
        if self.attention:
            refine_2 = nn.self_attention(refine_2, qk_chns=2*self.nr_filters, v_chns=2*self.nr_filters)
        refine_3 = RefineBlock(nr_filters=1*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, out_shape=jnp.shape(x_level_2_1)[0:3])([x_level_2_1, refine_2], h=proj_t)
        refine_4 = RefineBlock(nr_filters=1*self.nr_filters, normalizer=self.normalizer, nonlinearity=self.nonlinearity, out_shape=jnp.shape(x_level_1_1)[0:3], end=True)([x_level_1_1, refine_3], h=proj_t)
        

        out = self.normalizer()(refine_4)
        out = self.nonlinearity(out)
        out = nn.Conv(features=self.chns, kernel_size=(3,3))(out)
        
        if self.scale_out:
            out = out / t[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
        return out

if __name__ == '__main__':
    print('test')
    net = small(chns=2, nr_filters=64, affine_x=False, fourier_scale=32, attention=False, dropout=0., scale_out=True, normalizer=InstanceNorm2dPlus, nonlinearity=nn.elu)
    tabulate_fn = nn.tabulate(net, jax.random.PRNGKey(0))
    print(tabulate_fn(x=jnp.ones((10, 256, 256, 2)), t=jnp.ones((10))))