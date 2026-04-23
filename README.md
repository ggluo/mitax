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

## Getting Started

Check out the `examples/` directory for sample scripts:

- **MNIST Diffusion**: Run `python examples/mnist/mnist_classifier.py` or explore `examples/mnist/mnist.py`.
- **Brain MRI**: See `examples/brain/train.py` for a training pipeline example.

## License

This project is licensed under the MIT License - see the `LICENSE.txt` file for details.
