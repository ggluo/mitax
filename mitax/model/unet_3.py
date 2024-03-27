import functools
import flax.linen as nn
import jax.numpy as jnp

from mitax.misc import utils
from mitax.model.module2 import *
import einops

# causal unet
class ncsnpp(nn.Module):

  dropout:          float = 0.1
  scale_out:        bool = True
  nonlinearity:     str = 'swish'
  nr_filters:       int = 128
  ch_mult:          tuple = (1, 2, 2, 2)
  nr_res_blocks:    int = 4
  attn_resolutions: tuple = (16,)
  resamp_with_conv: bool = True
  fir:              bool = True
  fir_kernel:       list = (1, 3, 3, 1)
  skip_rescale:     bool = True
  resblock_type:    str = 'biggan'

  progressive:         str = 'none'
  progressive_input:   str = 'residual'
  progressive_combine: str = 'sum'

  init_scale:    float = 0.
  fourier_scale: int = 16
  center_x:      bool = False

  seq_len: int = 5

  @nn.compact
  def __call__(self, x0, x1, t, train=True):
    
    # x0 contains the previous frames 
    # x1 contains the current frame
    # scaling the input

    nonlinearity = utils.get_class_by_name(nn, self.nonlinearity)
    num_resolutions = len(self.ch_mult)

    assert self.progressive in ['none', 'output_skip', 'residual']
    assert self.progressive_input in ['none', 'input_skip', 'residual']
    combiner = functools.partial(Combine, method=self.progressive_combine)

    used_sigmas = t
    temb = GaussianFourierProjection(embedding_size=self.nr_filters, scale=self.fourier_scale)(jnp.log(used_sigmas))

    temb = nn.Dense(self.nr_filters * 4, kernel_init=default_init())(temb)
    temb = nn.Dense(self.nr_filters * 4, kernel_init=default_init())(nonlinearity(temb))
   

    AttnBlock = functools.partial(AttnBlockpp, init_scale=self.init_scale, skip_rescale=self.skip_rescale)

    Upsampler = functools.partial(Upsample, with_conv=self.resamp_with_conv, fir=self.fir, fir_kernel=self.fir_kernel)
    if self.progressive == 'output_skip':
      pyramid_upsample = functools.partial(Upsample, fir=self.fir, fir_kernel=self.fir_kernel, with_conv=False)
    elif self.progressive == 'residual':
      pyramid_upsample = functools.partial(Upsample, fir=self.fir, fir_kernel=self.fir_kernel, with_conv=True)

    Downsampler = functools.partial(Downsample, with_conv=self.resamp_with_conv, fir=self.fir, fir_kernel=self.fir_kernel)
    if self.progressive_input == 'input_skip':
      pyramid_downsample = functools.partial(Downsample, fir=self.fir, fir_kernel=self.fir_kernel, with_conv=False)
    elif self.progressive_input == 'residual':
      pyramid_downsample = functools.partial(Downsample,fir=self.fir, fir_kernel=self.fir_kernel, with_conv=True)

    if self.resblock_type == 'ddpm':
      ResnetBlock = functools.partial(ResnetBlockDDPMpp,
                                      act=nonlinearity,
                                      dropout=self.dropout,
                                      init_scale=self.init_scale,
                                      skip_rescale=self.skip_rescale)

    elif self.resblock_type == 'biggan':
      ResnetBlock = functools.partial(ResnetBlockBigGANpp,
                                      act=nonlinearity,
                                      dropout=self.dropout,
                                      fir=self.fir,
                                      fir_kernel=self.fir_kernel,
                                      init_scale=self.init_scale,
                                      skip_rescale=self.skip_rescale)

    else:
      raise ValueError("resblock type %s unrecognized."%self.resblock_type)


    # check if preprocessing sequence of images
    assert len(x0.shape) == 5
    
    b_x, t_x, h_x, w_x, c_x = x0.shape
    x0 = einops.rearrange(x0, 'b t h w c -> (b t) h w c')
    x0 = conv3x3(x0, self.nr_filters)

    temb = einops.repeat(temb, 'b c -> (b t) c', t=t_x)
    x1 = einops.rearrange(x1, 'b t h w c -> (b t) h w c')

    # Downsampling block
    input_pyramid = None
    if self.progressive_input != 'none':
      input_pyramid = x1
    causal_mask = generate_causal_mask(self.seq_len)
    bst_x, sh_x, sw_x, sc_x = x0.shape
    st_x = bst_x//b_x
    
    hs = [conv3x3(x1, self.nr_filters)]
    xs0 = []
    for i_level in range(num_resolutions):
      
      t_nr = self.nr_filters*self.ch_mult[i_level-1] if i_level!=0 else self.nr_filters
      x0 = einops.rearrange(x0, '(b t) h w c -> (b h w) t c', b=b_x, h=sh_x, w=sw_x, t=st_x, c=sc_x)
      x0 = nn.LayerNorm()(x0)
      x0 = self_causal_attn(t_nr, causal_mask)(x0)
      x0 = einops.rearrange(x0, '(b h w) t c -> (b t) h w c', b=b_x, h=sh_x, w=sw_x, t=st_x, c=t_nr)
      x0 = ResnetBlock(out_ch=t_nr, down=False if i_level==0 else True)(x0, temb, train)

      bst_x, sh_x, sw_x, sc_x = x0.shape
      st_x = bst_x//b_x
      xs0.append(x0)
      hs[-1] = hs[-1] + x0

      for i_block in range(self.nr_res_blocks):
        h = ResnetBlock(out_ch=self.nr_filters * self.ch_mult[i_level])(hs[-1], temb, train)
        if h.shape[1] in self.attn_resolutions:
          h = AttnBlock()(h)
        hs.append(h)

      if i_level != num_resolutions - 1:
        if self.resblock_type == 'ddpm':
          h = Downsampler()(hs[-1])
        else:
          h = ResnetBlock(down=True)(hs[-1], temb, train)

        if self.progressive_input == 'input_skip':
          input_pyramid = pyramid_downsample()(input_pyramid)
          h = combiner()(input_pyramid, h)

        elif self.progressive_input == 'residual':
          input_pyramid = pyramid_downsample(out_ch=h.shape[-1])(input_pyramid)
          if self.skip_rescale:
            input_pyramid = (input_pyramid + h) / np.sqrt(2.)
          else:
            input_pyramid = input_pyramid + h
          h = input_pyramid

        hs.append(h)

    h = hs[-1]
    h = ResnetBlock()(h, temb, train)
    h = AttnBlock()(h)
    h = ResnetBlock()(h, temb, train)

    pyramid = None

    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(self.nr_res_blocks + 1):
        if i_block == self.nr_res_blocks:
          h = ResnetBlock(out_ch=self.nr_filters * self.ch_mult[i_level])(jnp.concatenate([h, hs.pop(), xs0.pop()], axis=-1), temb, train)
        else:
          h = ResnetBlock(out_ch=self.nr_filters * self.ch_mult[i_level])(jnp.concatenate([h, hs.pop()], axis=-1), temb, train)
      
      if h.shape[1] in self.attn_resolutions:
        h = AttnBlock()(h)

      if self.progressive != 'none':
        if i_level == num_resolutions - 1:
          if self.progressive == 'output_skip':
            pyramid = conv3x3(
              nonlinearity(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h)),
              x.shape[-1],
              bias=True,
              init_scale=self.init_scale)
          elif self.progressive == 'residual':
            pyramid = conv3x3(
              nonlinearity(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h)),
              h.shape[-1],
              bias=True)
          else:
            raise ValueError("%s is not a valid name."%self.progressive)
        else:
          if self.progressive == 'output_skip':
            pyramid = pyramid_upsample()(pyramid)
            pyramid = pyramid + conv3x3(
              nonlinearity(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h)),
              x.shape[-1],
              bias=True,
              init_scale=self.init_scale)
          elif self.progressive == 'residual':
            pyramid = pyramid_upsample(out_ch=h.shape[-1])(pyramid)
            if self.skip_rescale:
              pyramid = (pyramid + h) / np.sqrt(2.)
            else:
              pyramid = pyramid + h
            h = pyramid
          else:
            raise ValueError("%s is not a valid name."%self.progressive)

      if i_level != 0:
        if self.resblock_type == 'ddpm':
          h = Upsampler()(h)
        else:
          h = ResnetBlock(up=True)(h, temb, train)

    assert not hs
    assert not xs0

    if self.progressive == 'output_skip':
      h = pyramid
    else:
      h = nonlinearity(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h))
      h = conv3x3(h, c_x, init_scale=self.init_scale)

    if self.scale_out:# not working for sequence of images
      used_sigmas = einops.repeat(used_sigmas, 'b -> (b t) 1 1 1', t=t_x)
      h = h / used_sigmas
    
    h = einops.rearrange(h, '(b t) h w c -> b t h w c', b=b_x, h=h_x, w=w_x, t=t_x, c=c_x)
    return h

if __name__ == '__main__':

  seq_len = 10
  net = ncsnpp(seq_len=seq_len, scale_out=False)
  input_shape = (1, seq_len+1, 64, 64, 1)
  x = jnp.zeros(input_shape)
  x0 = x[:, :-1, ...]
  x1 = x[:, 1:, ...]

  t = jnp.ones((input_shape[0]), dtype=jnp.float32)
  params = net.init(jax.random.PRNGKey(0), x0, x1, t, train=False)
  out    = net.apply(params, x0, x1, t, train=True, rngs={'dropout': jax.random.PRNGKey(0)})
  l2     = jnp.mean(out-x1, axis=(-3,-2,-1))
  print(l2)

  x0 = x0.at[:, 4, ...].set(1)
  out    = net.apply(params, x0, x1, t, train=True, rngs={'dropout': jax.random.PRNGKey(0)})
  l2     = jnp.mean(out-x1, axis=(-3,-2,-1))
  print(l2)


