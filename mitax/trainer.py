from typing import Any
from flax.training import train_state
from flax.metrics import tensorboard
from collections import defaultdict
import optax

from pathlib import Path
import os



class trainer:

    def __init__(self,
                 model_name : str,
                 model_hparams: dict,
                 optimizer_name : str,
                 optimizer_hparams : dict,
                 init_input : Any,
                 seed=1234):
        
        super().__init__()

        self.model_name        = model_name
        self.model_hparams     = model_hparams
        self.optimizer_name    = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.seed              = seed

        self.model             = self.init_model(self.model_name, self.model_hparams)
        self.logdir            = ""
        self.logger            = ""


    def create_train_state(self, rng, config):
        """creates initial train state"""

        return train_state.TrainState.create()


    def init_model(self, name, params):
        return None

    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        # init learning rate schedule and optimizer
        opt_name = self.optimizer_name.lower()
        if opt_name == 'adam':
            opt_class = optax.adam
        elif opt_name == 'adamw':
            opt_class = optax.adamw
        elif opt_name == 'sgd':
            opt_class = optax.sgd
        else:
            assert False, f'Unknown "optimizer_name"'

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
        # save the snapshot of model during the training
        pass

    def restore(self,):
        # restore the training state from checkpoint file
        pass

    def train_step(self, params, optimizer_state, batch, targets):
        # Write your training step logic here
        return new_params, new_optimizer_state

    def train_loop(self, dataloader):
        # loop for one epoch
        for step, batch in enumerate(dataloader):
            params, optimizer_state = self.train_step(params, optimizer_state, batch['data'], batch['targets'])

            if step % 100 == 0:  # You might want to log or print the training progress
                # Log training progress
                pass

    def test_model(self, dataloader):

        for batch in dataloader:
            pass

    def train(self, train_loader, test_loader, epochs, num_steps_per_epoch, snap_interval):

        self.init_model()
        self.create_train_state(self.init_optimizer())

        for epoch in range(epochs):

            self.train_loop()

            if (epoch+1) % snap_interval == 0:
                self.save_snapshot()