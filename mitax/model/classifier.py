from typing import Any
import optax

class cross_entropy():

    def __init__(self, **loss_params):
        self.loss_params = loss_params

    
    def __call__(self, apply_fn, params, batch, training):
        imgs, labels, key = batch
        logits = apply_fn({'params': params},
                                    imgs,
                                    train=training,
                                    rngs={'dropout': key} if training else None)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        accuracy = (logits.argmax(axis=-1) == labels).mean()
        return loss, {'accuracy': accuracy}
        