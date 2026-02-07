from __future__ import annotations
from typing import Callable, Tuple
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers

def _spike_and_slab(event_shape, dtype):
    return tfd.Mixture(
        cat=tfd.Categorical(probs=[0.5, 0.5]),
        components=[
            tfd.Independent(
                tfd.Normal(loc=tf.zeros(event_shape, dtype=dtype),
                           scale=1.0 * tf.ones(event_shape, dtype=dtype)),
                reinterpreted_batch_ndims=1,
            ),
            tfd.Independent(
                tfd.Normal(loc=tf.zeros(event_shape, dtype=dtype),
                           scale=10.0 * tf.ones(event_shape, dtype=dtype)),
                reinterpreted_batch_ndims=1,
            ),
        ],
        name="spike_and_slab",
    )

def make_prior_fn():
    def _prior(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfpl.DistributionLambda(lambda _: _spike_and_slab(n, dtype))
        ])
    return _prior

def make_posterior_fn():
    def _posterior(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
            tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
            tfpl.IndependentNormal(n),
        ])
    return _posterior

def negative_log_likelihood(y_true, y_pred):
    return -y_pred.log_prob(y_true)

def make_bnn_classifier(input_shape: Tuple[int, ...], num_classes: int) -> tf.keras.Model:
    """BNN classifier using variational layers (TFP)."""
    prior_fn = make_prior_fn()
    post_fn = make_posterior_fn()

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tfpl.DenseVariational(
            units=128,
            make_prior_fn=prior_fn,
            make_posterior_fn=post_fn,
            kl_weight=1.0,  # set in compile via model.losses scaling if needed
            activation="relu",
        ),
        tfpl.DenseVariational(
            units=num_classes,
            make_prior_fn=prior_fn,
            make_posterior_fn=post_fn,
            kl_weight=1.0,
        ),
        tfpl.OneHotCategorical(num_classes),
    ])
    return model
