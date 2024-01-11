import sys
sys.path.append('/home/gluo/mitax')

import functools
import flax.linen as nn
import jax.numpy as jnp

from mitax.misc import utils
from mitax.model.module2 import *


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

  @nn.compact
  def __call__(self, x, t, train=True):
    

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

    if self.center_x:
        x = 2 * x - 1.

    # Downsampling block
    input_pyramid = None
    if self.progressive_input != 'none':
      input_pyramid = x

    hs = [conv3x3(x, self.nr_filters)]
    for i_level in range(num_resolutions):
      
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

    if self.progressive == 'output_skip':
      h = pyramid
    else:
      h = nonlinearity(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h))
      h = conv3x3(h, x.shape[-1], init_scale=self.init_scale)

    if self.scale_out:
      used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
      h = h / used_sigmas

    return h

if __name__ == '__main__':

  net = ncsnpp()
  tabulate_fn = nn.tabulate(net, jax.random.PRNGKey(0))
  print(tabulate_fn(x=jnp.ones((10, 256, 256, 2)), t=jnp.ones((10)), train=False))