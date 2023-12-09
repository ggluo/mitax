from typing import Any
from flax.training import train_state, orbax_utils
from flax.metrics.tensorboard import SummaryWriter
import optax
import orbax.checkpoint as ocp
import jax
import time

from mitax.misc import utils

class trainer:
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
                 loss_name: str,
                 loss_params: dict,
                 optimizer_name: str,
                 optimizer_hparams: dict,
                 init_input: Any,
                 logdir: str,
                 seed=1234):
            """
            Initializes the Trainer class.

            Args:
                model_name (str): The module and class name of the model.
                model_hparams (dict): Hyperparameters for the model.
                optimizer_name (str): The name of the optimizer.
                optimizer_hparams (dict): Hyperparameters for the optimizer.
                init_input (Any): The initial input for the trainer.
                logdir (str): The directory path to save logs.
                seed (int, optional): The seed value for random number generation. Defaults to 1234.
            """
            super().__init__()

            self.model_name        = model_name
            self.model_hparams     = model_hparams
            self.loss_name         = loss_name
            self.loss_params       = loss_params
            self.optimizer_name    = optimizer_name
            self.optimizer_hparams = optimizer_hparams
            self.seed              = seed

            self.init_model(self.model_name, self.model_hparams, init_input)
            self.logdir = utils.create_folder(logdir)
            self.writer = SummaryWriter(self.logdir)
            self.orbax_checkpointer = ocp.PyTreeCheckpointer()

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
        module_name, class_name = name.rsplit('.', 1)

        self.model = utils.get_class_by_name(module_name, class_name)(**params)
        init_rng   = jax.random.PRNGKey(self.seed)
        variables  = self.model.init(init_rng, init_input, train=True)
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
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=self.optimizer_hparams.pop('lr'),
            boundaries_and_scales=
                {int(num_steps_per_epoch*num_epochs*0.6): 0.1,
                    int(num_steps_per_epoch*num_epochs*0.85): 0.1}
        )

        # Clip gradients at max value, and evt. apply weight decay
        transf = [optax.clip(1.0)]

        if opt_class == optax.sgd and 'weight_decay' in self.optimizer_hparams: 
            transf.append(optax.add_decayed_weights(self.optimizer_hparams.pop('weight_decay')))

        return optax.chain(*transf,opt_class(lr_schedule, **self.optimizer_hparams))

    def save_model(self, step):
        """
        Saves a snapshot of the model during the training process.

        Returns:
            None
        """
        ckpt = {'model': self.state}
        save_args = orbax_utils.save_args_from_target(ckpt)
        try:
            self.orbax_checkpointer.save(self.logdir+ '/' + self.model_name + '_' + str(step), ckpt, save_args=save_args)

        except:
            raise ValueError('Failed to save model')

    def load_model(self, path):
        """
        Restores the training state from a checkpoint file.

        Returns:
            the restored training state.
        """
        try:
            a = self.orbax_checkpointer.restore(path)
            return a
        except:
            raise ValueError('Failed to load model')

    def create_functions(self):
     
        def init_loss(name, model, params, loss_params, batch, train):
            module_name, class_name = name.rsplit('.', 1)
            loss = utils.get_class_by_name(module_name, class_name)(loss_params)
            return loss(model, params, batch, train)
        
        def train_step(state, batch):
            loss_fn = lambda params: init_loss(self.loss_name, self.model, params, self.loss_params, batch, train=True)
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params) # ret[0] is loss, ret[1] is metric
            ret[1]['loss'] = ret[0]
            state = state.apply_gradients(grads=grads)
            return state, ret[1]

        def test_step(state, batch):
            loss, metric = init_loss(self.loss_name, self.model, state.params, self.loss_params, batch, train=True)
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
        print('Epoch {:d}: {} in {:0.2f} sec -> '.format(epoch, prefix, time), end='')
        for k, v in metric.items():
            print(k + ': {:0.4f}'.format(v), end=' ')
        print()

    def kernel(self, dataloader, train=True, log_interval=100):
        """
        The training loop for one epoch.

        Args:
            dataloader: The dataloader for training data.

        Returns:
            None
        """
        
        
        lm = []

        for inner_step, batch in enumerate(dataloader):

            if train:
                self.state, metric = self.train_step(self.state, batch)
                if inner_step % log_interval == 0:  # You might want to log or print the training progress
                    self.write_summary(self.state.step, metric, prefix='train')
            else:
                metric = self.test_step(self.state, batch)
            lm.append(metric)

        avg_metric = {k: sum(d[k] for d in lm) / len(lm) for k in lm[0]}

        if not train:
            self.write_summary(self.state.step, avg_metric, prefix='test')
        return avg_metric

    def train(self, train_loader, test_loader, epochs, num_steps_per_epoch, snap_interval, pretrained_model=None):
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

        if pretrained_model is not None:
            state_dict = self.load_model(pretrained_model)
            self.state = state_dict['model']

        self.state = train_state.TrainState.create(apply_fn = self.model.apply,
                                                params = self.init_params if self.state is None else self.state['params'],
                                                tx     = self.init_optimizer(epochs, num_steps_per_epoch))
        self.create_functions()

        for epoch in range(epochs):

            begin_t    = time.time()
            self.display_summary(epoch+1, self.kernel(train_loader, True), time.time()-begin_t, prefix='train')

            begin_t    = time.time()
            self.display_summary(epoch+1, self.kernel(test_loader, False), time.time()-begin_t, prefix='test' )

            if (epoch+1) % snap_interval == 0:
                self.save_model(epoch)