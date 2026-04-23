# MITAX: Medical Imaging Toolbox with JAX

MITAX is a JAX-based toolbox for medical imaging, with a primary focus on MRI reconstruction and generative models (specifically diffusion models). It leverages JAX and Flax for high-performance, differentiable medical imaging pipelines.

## Features

- **JAX-based MRI Operators**: Efficient implementations of MRI operations including:
    - Multi-coil FFT/IFFT.
    - ESPIRIT coil sensitivity estimation.
    - Cartesian SENSE-like iterative reconstruction kernels.
- **Generative Models**:
    - Score-based Generative Modeling (SMLD) implementations.
    - Support for various noise schedules and SDE-based sampling.
- **Flax Training Pipeline**:
    - A modular `Trainer` class for managing Flax-based models.
    - Integration with `optax` for optimizers and `orbax` for checkpointing.
    - Support for EMA (Exponential Moving Average) weights.
- **Dataflow Utilities**:
    - Asynchronous and parallel data loading and preprocessing utilities designed for medical imaging datasets.

## Project Structure

```text
mitax/
├── dataflow/    # Data loading and preprocessing utilities
├── model/       # Neural network architectures (UNet, RefineNet) and Diffusion Models
├── mri/         # MRI-specific operations (operators, coil estimation)
├── trainer.py   # General-purpose trainer for Flax models
├── sampler.py   # Sampling utilities for generative models
└── exporter.py  # Model export utilities
examples/
├── brain/       # MRI brain reconstruction examples
└── mnist/       # MNIST-based diffusion model examples
```

## Installation

You can install MITAX by cloning the repository and using `pip`:

```bash
git clone https://github.com/your-repo/mitax.git
cd mitax
pip install .
```

Requires Python >= 3.7.

## Demos and Examples

### Brain MRI Reconstruction
This example demonstrates how to use MITAX for MRI reconstruction using a pre-trained diffusion model. You can find the full demo in `examples/brain/demo.ipynb`.

#### Core Reconstruction Workflow
```python
import jax
import jax.numpy as jnp
from mitax.sampler import AncestralSampler
from mitax.misc import utils

# Load configuration and initialize sampler
config = utils.load_config('path/to/config.yaml')
sampler = AncestralSampler(
    net_name=config['net_name'],
    net_hparams=config['net_hparams'],
    dm_name=config['loss_name'],
    dm_hparams=config['loss_params'],
    target_snr=0.2,
    init_input={'x': jnp.ones((1, 256, 256, 2)), 't': jnp.ones((1))},
    path='path/to/model/weights'
)

# Perform reconstruction
input_shape = (1, 256, 256, 2)
sampler.create_functions()
_, reconstructed_image = sampler(
    jax.random.normal(jax.random.PRNGKey(0), input_shape) * sampler.dm.sigma_max
)
```

### MNIST Diffusion Model
The MNIST examples showcase score-based generative modeling (SMLD) on a simpler dataset.
- **Training**: `python examples/mnist/mnist.py`
- **Classifier-guided sampling**: `python examples/mnist/mnist_classifier.py`

### Running the Examples
To run the examples, ensure your `PYTHONPATH` includes the repository root:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
# Example: Training the Brain MRI model
python examples/brain/train.py examples/brain/smld_unet.yaml 0
```

## License

This project is licensed under the MIT License - see the `LICENSE.txt` file for details.
