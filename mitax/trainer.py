from typing import Any, TypeVar, Mapping
from flax.training import train_state, orbax_utils
from flax.metrics.tensorboard import SummaryWriter
import optax
import orbax.checkpoint as ocp
import jax
import time
from jax import tree_util
import jax.numpy as jnp
from mitax.misc import utils
import os
import torch

TX = TypeVar("TX", bound=optax.OptState)


def restore_optimizer_state(opt_state: TX, restored: Mapping[str, Any]) -> TX:
    """Restore optimizer state from loaded checkpoint (or .msgpack file)."""
    return tree_util.tree_unflatten(
        tree_util.tree_structure(opt_state), tree_util.tree_leaves(restored)
    )

class trainer:
    """
    The `Trainer` class is responsible for initializing the trainer and managing the training process.

    Args:
        net_name (str): The module and class name of the net.
        net_hparams (dict): Hyperparameters for the net.
        optimizer_name (str): The name of the optimizer.
        optimizer_hparams (dict): Hyperparameters for the optimizer.
        init_input (Any): The initial input for the trainer.
        seed (int, optional): The seed value for random number generation. Defaults to 1234.
    """

    def __init__(self,
                 net_name: str,
                 net_hparams: dict,
                 loss_name: str,
                 loss_params: dict,
                 optimizer_name: str,
                 optimizer_hparams: dict,
                 init_input: Any,
                 logdir: str,
                 rng_key: Any):
            """
            Initializes the Trainer class.

            Args:
                net_name (str): The module and class name of the net.
                net_hparams (dict): Hyperparameters for the net.
                optimizer_name (str): The name of the optimizer.
                optimizer_hparams (dict): Hyperparameters for the optimizer.
                init_input (Any): The initial input for the trainer.
                logdir (str): The directory path to save logs.
                rng_key (int, optional): for random funcs in dropout and diffusion
            """
            super().__init__()

            self.net_name        = net_name
            self.net_hparams     = net_hparams
            self.loss_name         = loss_name
            self.loss_params       = loss_params
            self.optimizer_name    = optimizer_name
            self.optimizer_hparams = optimizer_hparams
            self.rng_key           = rng_key

            self.init_net(self.net_name, self.net_hparams, init_input)
            if os.path.exists(logdir):
                self.logdir = logdir
            else:
                self.logdir = utils.create_folder(logdir, prefix=loss_name.split('.')[-1])
            self.writer = SummaryWriter(self.logdir)
            self.orbax_checkpointer = ocp.PyTreeCheckpointer()
            if 'adaptive' in loss_params.keys():
                self.adaptive_loss = loss_params.pop('adaptive')
            else:
                self.adaptive_loss = False

    def init_net(self, name, params, init_input):
        """
        Initializes the net with the given name, parameters, and input.

        Args:
            name (tuple): A tuple containing the module name and class name of the net.
            params (dict): A dictionary of parameters to be passed to the net constructor.
            init_input: The input used for initializing the net.

        Returns:
            None
        """
        module_name, class_name = name.rsplit('.', 1)

        self.net = utils.get_class_by_name(module_name, class_name)(**params)
        self.rng_key, init_rng = jax.random.split(self.rng_key)
        variables              = self.net.init(init_rng, **init_input, train=False)# jitize the net.init
        self.init_params       = variables['params']
        self.state             = None

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
        #lr_schedule = optax.piecewise_constant_schedule(
        #    init_value=self.optimizer_hparams.pop('lr'),
        #    boundaries_and_scales=
        #        {int(num_steps_per_epoch*num_epochs*0.6): 0.1,
        #            int(num_steps_per_epoch*num_epochs*0.85): 0.1}
        #)

        # Clip gradients at max value, and evt. apply weight decay
        # transf = [optax.clip(1.0)]

        #if opt_class == optax.sgd and 'weight_decay' in self.optimizer_hparams: 
        #    transf.append(optax.add_decayed_weights(self.optimizer_hparams.pop('weight_decay')))

        if 'ema_decay' in self.optimizer_hparams.keys():
            ema_decay = self.optimizer_hparams.pop('ema_decay')
            return optax.chain(opt_class(self.optimizer_hparams.pop('lr'), **self.optimizer_hparams),
                               optax.ema(ema_decay))

        else:
            return opt_class(self.optimizer_hparams.pop('lr'), **self.optimizer_hparams)

    def save_model(self, step):
        """
        Saves a snapshot of the model during the training process.

        Returns:
            None
        """
        ckpt = {'model': self.state}
        save_args = orbax_utils.save_args_from_target(ckpt)
        try:
            self.orbax_checkpointer.save(self.logdir+ '/' + self.net_name + '_' + str(step), ckpt, save_args=save_args)

        except:
            raise ValueError('Failed to save model')

    def load_model(self, path):
        """
        Restores the training state from a checkpoint file.

        Returns:
            the restored training state.
        """
        try:
            return self.orbax_checkpointer.restore(path)
        except:
            raise ValueError('Failed to load model')

    def create_functions(self, total_d=None):
        # create train step function and test step function

        def init_loss(name, apply_fn, params, loss_params, batch, train):
            module_name, class_name = name.rsplit('.', 1)
            if self.adaptive_loss:
                loss_params['total_d'] = total_d
            loss = utils.get_class_by_name(module_name, class_name)(**loss_params)
            return loss(apply_fn, params, batch, train)
        
        def train_step(state, batch):
            loss_fn = lambda params, batch: init_loss(self.loss_name, self.state.apply_fn, params, self.loss_params, batch, train=True)
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch) # ret[0] is loss, ret[1] is metric
            ret[1]['loss'] = ret[0]
            state = state.apply_gradients(grads=grads)
            return state, ret[1]

        def test_step(state, batch):
            loss_fn = lambda params, batch: init_loss(self.loss_name, self.state.apply_fn, params, self.loss_params, batch, train=False)
            loss, metric = loss_fn(state.params, batch)
            metric['loss'] = loss
            return metric
        
        self.train_step = jax.jit(train_step)
        self.test_step = jax.jit(test_step)

    def write_summary(self, step, metric, prefix='train'):
        """
        Writes the summary of the training process.

        Args:
            step (int): The current training step.
            metric (dict): The current training metric.

        Returns:
            None
        """
        
        for k, v in metric.items():
            self.writer.scalar(prefix + '_' + k, v, step)
    
    def display_summary(self, epoch, metric, time, prefix='train'):
        """
        Writes the summary of the training process.

        Args:
            step (int): The current training step.
            metric (dict): The current training metric.

        Returns:
            None
        """
        if metric is not None:
            print('Epoch {:d}: {} in {:0.2f} sec -> '.format(epoch, prefix, time), end='')
            for k, v in metric.items():
                print(k + ': {:0.4f}'.format(v), end=' ')
            print()

    def kernel(self, dataloader, train=True, log_interval=2):
        """
        The training loop for one epoch.

        Args:
            dataloader: The dataloader for training data.

        Returns:
            None
        """
        
        
        lm = []
        avg_metric = None
        for inner_step, batch in enumerate(dataloader):

            if not isinstance(batch, (tuple, list)):
                batch = [batch,]

            # in case of torch tensors, convert to jax arrays
            for ix, b in enumerate(batch):
                if isinstance(b, torch.Tensor):
                    batch[ix] = jnp.array(b.numpy())

            # key for dropout and diffusion
            self.rng_key, subkey = jax.random.split(self.rng_key)
            batch.append(subkey)

            if self.adaptive_loss:
                batch.append(self.state.step)

            if train:
                self.state, metric = self.train_step(self.state, batch)
                if inner_step % log_interval == 0:  # You might want to log or print the training progress
                    self.write_summary(self.state.step, metric, prefix='train')
            else:
                metric = self.test_step(self.state, batch)
            lm.append(metric)

        if len(lm) > 0:
            avg_metric = {k: sum(d[k] for d in lm) / len(lm) for k in lm[0]}

        if not train:
            self.write_summary(self.state.step, avg_metric, prefix='test')
        return avg_metric

    def train(self, train_loader, test_loader, epochs, num_steps_per_epoch, snap_interval):
        """
        Trains the model using the specified train and test loaders, number of epochs, number of steps per epoch, and snapshot interval.

        Args:
            train_loader: The dataloader for training data.
            test_loader: The dataloader for test data.
            epochs (int): Number of epochs for training.
            num_steps_per_epoch (int): Number of steps per epoch.
            snap_interval (int): Interval for saving snapshots of the model.
            restore_model (str, optional): The path to a logging folder for restoring the model.
        Returns:
            None
        """        

        self.state = train_state.TrainState.create(apply_fn = self.net.apply,
                                                params = self.init_params,
                                                tx     = self.init_optimizer(epochs, num_steps_per_epoch))

        base_e = 0
        r_path = utils.get_last_folder(self.logdir)
        if r_path is not None:
            base_e = int(r_path.split('_')[-1])
            state_dict = self.load_model(os.path.join(self.logdir, r_path))['model']
            restored_optimizer = restore_optimizer_state(self.state.opt_state, state_dict["opt_state"])
            self.state = self.state.replace(step=state_dict['step'], params=state_dict['params'], opt_state=restored_optimizer)

        self.create_functions(epochs * num_steps_per_epoch) # epochs * num for adaptive loss

        for epoch in range(epochs-base_e):

            begin_t    = time.time()
            self.display_summary(base_e+epoch+1, self.kernel(train_loader, True), time.time()-begin_t, prefix='train')

            begin_t    = time.time()
            self.display_summary(base_e+epoch+1, self.kernel(test_loader, False), time.time()-begin_t, prefix='test' )

            if (epoch+1) % snap_interval == 0:
                self.save_model(base_e+epoch+1)