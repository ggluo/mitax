from typing import Any
import optax

class cross_entropy():

    def __init__(self, loss_params: dict):
        self.loss_params = loss_params

    
    def __call__(self, model, params, batch, training):
        imgs, labels = batch
        logits = model.apply({'params': params},
                                    imgs,
                                    train=training)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        accuracy = (logits.argmax(axis=-1) == labels).mean()
        return loss, {'accuracy': accuracy}
        