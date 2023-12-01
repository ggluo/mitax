import jax
from flax import linen as nn
from registry import registered_classes


class mnist_cnn(nn.Module):
    """A simple cnn model for mnist classifying"""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2,2), strides=(2,2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x

registered_classes['mnist_cnn'] = mnist_cnn