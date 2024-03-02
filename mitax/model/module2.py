import flax.linen as nn
import jax.nn as jnn
import jax.numpy as jnp
import jax
import numpy as np
from typing import Any, Optional, Tuple

import string
from mitax.model import up_or_down_sampling

def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return jnn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')

def conv1x1(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
  """1x1 convolution with DDPM initialization."""
  bias_init = jnn.initializers.zeros
  output = nn.Conv(out_planes, kernel_size=(1, 1),
                   strides=(stride, stride), padding='SAME', use_bias=bias,
                   kernel_dilation=(dilation, dilation),
                   kernel_init=default_init(init_scale),
                   bias_init=bias_init)(x)
  return output


def conv3x3(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
  """3x3 convolution with DDPM initialization."""
  bias_init = jnn.initializers.zeros
  output = nn.Conv(
    out_planes,
    kernel_size=(3, 3),
    strides=(stride, stride),
    padding='SAME',
    use_bias=bias,
    kernel_dilation=(dilation, dilation),
    kernel_init=default_init(init_scale),
    bias_init=bias_init)(x)
  return output

class NIN(nn.Module):
  num_units: int
  init_scale: float = 0.1

  @nn.compact
  def __call__(self, x):
    in_dim = int(x.shape[-1])
    W = self.param('W', default_init(scale=self.init_scale), (in_dim, self.num_units))
    b = self.param('b', jnn.initializers.zeros, (self.num_units,))
    y = contract_inner(x, W) + b
    assert y.shape == x.shape[:-1] + (self.num_units,)
    return y
  
def _einsum(a, b, c, x, y):
  einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
  return jnp.einsum(einsum_str, x, y)


def contract_inner(x, y):
  """tensordot(x, y, 1)."""
  x_chars = list(string.ascii_lowercase[:len(x.shape)])
  y_chars = list(string.ascii_uppercase[:len(y.shape)])
  assert len(x_chars) == len(x.shape) and len(y_chars) == len(y.shape)
  y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
  out_chars = x_chars[:-1] + y_chars[1:]
  return _einsum(x_chars, y_chars, out_chars, x, y)

class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""
  embedding_size: int = 256
  scale: float = 1.0

  @nn.compact
  def __call__(self, x):
    W = self.param('W', jax.nn.initializers.normal(stddev=self.scale), (self.embedding_size,))
    W = jax.lax.stop_gradient(W)
    x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
    return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)

class Combine(nn.Module):
  """Combine information from skip connections."""
  method: str = 'cat'

  @nn.compact
  def __call__(self, x, y):
    h = conv1x1(x, y.shape[-1])
    if self.method == 'cat':
      return jnp.concatenate([h, y], axis=-1)
    elif self.method == 'sum':
      return h + y
    else:
      raise ValueError(f'Method {self.method} not recognized.')

class AttnBlockpp(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""
  skip_rescale: bool = False
  init_scale: float = 0.

  @nn.compact
  def __call__(self, x):
    B, H, W, C = x.shape
    h = nn.GroupNorm(num_groups=min(x.shape[-1] // 4, 32))(x)
    q = NIN(C)(h)
    k = NIN(C)(h)
    v = NIN(C)(h)

    w = jnp.einsum('bhwc,bHWc->bhwHW', q, k) * (int(C) ** (-0.5))
    w = jnp.reshape(w, (B, H, W, H * W))
    w = jax.nn.softmax(w, axis=-1)
    w = jnp.reshape(w, (B, H, W, H, W))
    h = jnp.einsum('bhwHW,bHWc->bhwc', w, v)
    h = NIN(C, init_scale=self.init_scale)(h)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)


class Upsample(nn.Module):
  out_ch: Optional[int] = None
  with_conv: bool = False
  fir: bool = False
  fir_kernel: Tuple[int] = (1, 3, 3, 1)

  @nn.compact
  def __call__(self, x):
    B, H, W, C = x.shape
    out_ch = self.out_ch if self.out_ch else C
    if not self.fir:
      h = jax.image.resize(x, (x.shape[0], H * 2, W * 2, C), 'nearest')
      if self.with_conv:
        h = conv3x3(h, out_ch)
    else:
      if not self.with_conv:
        h = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.Conv2d(out_ch,
                                       kernel=3,
                                       up=True,
                                       resample_kernel=self.fir_kernel,
                                       use_bias=True,
                                       kernel_init=default_init())(x)

    assert h.shape == (B, 2 * H, 2 * W, out_ch)
    return h

class Downsample(nn.Module):
  out_ch: Optional[int] = None
  with_conv: bool = False
  fir: bool = False
  fir_kernel: Tuple[int] = (1, 3, 3, 1)

  @nn.compact
  def __call__(self, x):
    B, H, W, C = x.shape
    out_ch = self.out_ch if self.out_ch else C
    if not self.fir:
      if self.with_conv:
        x = conv3x3(x, out_ch, stride=2)
      else:
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')
    else:
      if not self.with_conv:
        x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      else:
        x = up_or_down_sampling.Conv2d(
          out_ch,
          kernel=3,
          down=True,
          resample_kernel=self.fir_kernel,
          use_bias=True,
          kernel_init=default_init())(x)

    assert x.shape == (B, H // 2, W // 2, out_ch)
    return x


class ResnetBlockDDPMpp(nn.Module):
  """ResBlock adapted from DDPM."""
  act: Any
  out_ch: Optional[int] = None
  conv_shortcut: bool = False
  dropout: float = 0.1
  skip_rescale: bool = False
  init_scale: float = 0.

  @nn.compact
  def __call__(self, x, temb=None, train=True):
    B, H, W, C = x.shape
    out_ch = self.out_ch if self.out_ch else C
    h = self.act(nn.GroupNorm(num_groups=min(x.shape[-1] // 4, 32))(x))
    h = conv3x3(h, out_ch)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += nn.Dense(out_ch, kernel_init=default_init())(self.act(temb))[:, None, None, :]

    h = self.act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h))
    h = nn.Dropout(self.dropout)(h, deterministic=not train)
    h = conv3x3(h, out_ch, init_scale=self.init_scale)
    if C != out_ch:
      if self.conv_shortcut:
        x = conv3x3(x, out_ch)
      else:
        x = NIN(out_ch)(x)

    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)
    
class ResnetBlockBigGANpp(nn.Module):
  """ResBlock adapted from BigGAN."""
  act: Any
  up: bool = False
  down: bool = False
  out_ch: Optional[int] = None
  dropout: float = 0.1
  fir: bool = False
  fir_kernel: Tuple[int] = (1, 3, 3, 1)
  skip_rescale: bool = True
  init_scale: float = 0.

  @nn.compact
  def __call__(self, x, temb=None, train=True):
    B, H, W, C = x.shape
    out_ch = self.out_ch if self.out_ch else C
    h = self.act(nn.GroupNorm(num_groups=min(x.shape[-1] // 4, 32))(x))

    if self.up:
      if self.fir:
        h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
        x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
        x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
    elif self.down:
      if self.fir:
        h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
        x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
        x = up_or_down_sampling.naive_downsample_2d(x, factor=2)

    h = conv3x3(h, out_ch)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += nn.Dense(out_ch, kernel_init=default_init())(self.act(temb))[:, None, None, :]

    h = self.act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h))
    h = nn.Dropout(self.dropout)(h, deterministic=not train)
    h = conv3x3(h, out_ch, init_scale=self.init_scale)
    if C != out_ch or self.up or self.down:
      x = conv1x1(x, out_ch)

    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)