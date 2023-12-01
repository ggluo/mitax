from typing import Any
from flax.training import train_state
from flax.metrics import tensorboard
from collections import defaultdict
import optax
import jax

from pathlib import Path
import os


from mitax.misc import utils

class Trainer:
    """
    The `Trainer` class is responsible for initializing the trainer and managing the training process.

    Args:
        model_name (str): The module and class name of the model.
        model_hparams (dict): Hyperparameters for the model.
        optimizer_name (str): The name of the optimizer.
        optimizer_hparams (dict): Hyperparameters for the optimizer.
        init_input (Any): The initial input for the trainer.
        seed (int, optional): The seed value for random number generation. Defaults to 1234.
    """

    def __init__(self,
                    model_name: str,
                    model_hparams: dict,
                    optimizer_name: str,
                    optimizer_hparams: dict,
                    init_input: Any,
                    seed=1234):
        """
        Initializes the Trainer class.

        Args:
            model_name (str): The module and class name of the model.
            model_hparams (dict): Hyperparameters for the model.
            optimizer_name (str): The name of the optimizer.
            optimizer_hparams (dict): Hyperparameters for the optimizer.
            init_input (Any): The initial input for the trainer.
            seed (int, optional): The seed value for random number generation. Defaults to 1234.
        """
        super().__init__()

        self.model_name        = model_name
        self.model_hparams     = model_hparams
        self.optimizer_name    = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.seed              = seed

        self.init_model(self.model_name, self.model_hparams, init_input)
        self.logdir = ""
        self.logger = ""

    def create_train_state(self, optimizer):
        """
        Creates the train state for the trainer.

        Args:
            optimizer (optax.GradientTransformation): The optimizer for the trainer.

        Returns:
            None
        """
        self.state = train_state.TrainState.create(apply_fn = self.model.apply,
                                                params   = self.init_params if self.state is None else self.state.params,
                                                batch_stats = self.init_batch_stats if self.state is None else self.state.batch_stats,
                                                tx = optimizer)

    def init_model(self, name, params, init_input):
        """
        Initializes the model with the given name, parameters, and input.

        Args:
            name (tuple): A tuple containing the module name and class name of the model.
            params (dict): A dictionary of parameters to be passed to the model constructor.
            init_input: The input used for initializing the model.

        Returns:
            None
        """
        self.model = utils.get_registered_class(name)(**params)
        init_rng   = jax.random.PRNGKey(self.seed)
        variables  = self.model.init(init_rng, init_input, train=True)
        self.init_params       = variables['params']
        self.init_batch_stats  = variables['batch_stats']
        self.state             = None

    def init_loss(self, loss_name):
        """
        Initializes the loss function with the given name.

        Args:
            loss_name (str): The name of the loss function.

        Returns:
            None
        """
        
        loss_fn = utils.get_class_by_name('flax.metrics', loss_name)

    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        """
        Initialize the optimizer with a learning rate schedule and optional transformations.

        Args:
            num_epochs (int): Number of epochs for training.
            num_steps_per_epoch (int): Number of steps per epoch.

        Returns:
            optax.GradientTransformation: The initialized optimizer with the specified learning rate schedule and transformations.
        """
        # init learning rate schedule and optimizer
        opt_name = self.optimizer_name.lower()
        opt_class = utils.get_class_by_name('optax', opt_name)

        # decrease the learning rate by a factor of 0.1 after 60% and 85% of the training
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=self.optimizer_hparams['lr'],
            boundaries_and_scales=
                {int(num_steps_per_epoch*num_epochs*0.6): 0.1,
                    int(num_steps_per_epoch*num_epochs*0.85): 0.1}
        )

        # Clip gradients at max value, and evt. apply weight decay
        transf = [optax.clip(1.0)]

        if opt_class == optax.sgd and 'weight_decay' in self.optimizer_hparams: 
            transf.append(optax.add_decayed_weights(self.optimizer_hparams.pop('weight_decay')))

        return optax.chain(*transf,opt_class(lr_schedule, **self.optimizer_hparams))

    def save_snapshot(self,):
        """
        Saves a snapshot of the model during the training process.

        Returns:
            None
        """
        pass

    def restore(self,):
        """
        Restores the training state from a checkpoint file.

        Returns:
            None
        """
        pass

    def train_loop(self, dataloader):
        """
        The training loop for one epoch.

        Args:
            dataloader: The dataloader for training data.

        Returns:
            None
        """
        for step, batch in enumerate(dataloader):
            params, optimizer_state = self.train_step(params, optimizer_state, batch['data'], batch['targets'])

            if step % 100 == 0:  # You might want to log or print the training progress
                # Log training progress
                pass

    def test_model(self, dataloader):
        """
        Tests the model using the specified dataloader.

        Args:
            dataloader: The dataloader for test data.

        Returns:
            None
        """
        for batch in dataloader:
            pass

    def train(self, train_loader, test_loader, epochs, num_steps_per_epoch, snap_interval):
        """
        Trains the model using the specified train and test loaders, number of epochs, number of steps per epoch, and snapshot interval.

        Args:
            train_loader: The dataloader for training data.
            test_loader: The dataloader for test data.
            epochs (int): Number of epochs for training.
            num_steps_per_epoch (int): Number of steps per epoch.
            snap_interval (int): Interval for saving snapshots of the model.

        Returns:
            None
        """
        self.create_train_state(self.init_optimizer(epochs, num_steps_per_epoch))

        for epoch in range(epochs):
            self.train_loop(train_loader)
            self.test_model(test_loader) # You might want to log or print the test results

            if (epoch+1) % snap_interval == 0:
                self.save_snapshot()