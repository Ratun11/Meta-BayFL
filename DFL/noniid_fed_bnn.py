import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt

tfd = tfp.distributions
tfpl = tfp.layers

plt.rcParams['figure.figsize'] = (10, 6)

# Helper functions
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = 1 - x_train / 255.0
    x_test = 1 - x_test / 255.0
    y_train_oh = tf.keras.utils.to_categorical(y_train)
    y_test_oh = tf.keras.utils.to_categorical(y_test)

    return (x_train, y_train, y_train_oh), (x_test, y_test, y_test_oh)


def inspect_images(data, num_images):
    fig, ax = plt.subplots(nrows=1, ncols=num_images, figsize=(2 * num_images, 2))
    for i in range(num_images):
        # Reshape if necessary
        image = data[i].reshape(28, 28)
        ax[i].imshow(image, cmap='gray')
        ax[i].axis('off')
    plt.show()

def spike_and_slab(event_shape, dtype):
    distribution = tfd.Mixture(
        cat=tfd.Categorical(probs=[0.5, 0.5]),
        components=[
            tfd.Independent(tfd.Normal(
                loc=tf.zeros(event_shape, dtype=dtype),
                scale=1.0*tf.ones(event_shape, dtype=dtype)),
                            reinterpreted_batch_ndims=1),
            tfd.Independent(tfd.Normal(
                loc=tf.zeros(event_shape, dtype=dtype),
                scale=10.0*tf.ones(event_shape, dtype=dtype)),
                            reinterpreted_batch_ndims=1)],
    name='spike_and_slab')
    return distribution

def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)
def get_prior(kernel_size, bias_size, dtype=None):
    """
    This function should create the prior distribution, consisting of the
    "spike and slab" distribution that is described above.
    The distribution should be created using the kernel_size, bias_size and dtype
    function arguments above.
    The function should then return a callable, that returns the prior distribution.
    """
    n = kernel_size+bias_size
    prior_model = Sequential([tfpl.DistributionLambda(lambda t : spike_and_slab(n, dtype))])
    return prior_model

def get_posterior(kernel_size, bias_size, dtype=None):
    """
    This function should create the posterior distribution as specified above.
    The distribution should be created using the kernel_size, bias_size and dtype
    function arguments above.
    The function should then return a callable, that returns the posterior distribution.
    """
    n = kernel_size + bias_size
    return Sequential([
        tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
        tfpl.IndependentNormal(n)
    ])

def get_dense_variational_layer(prior_fn, posterior_fn, kl_weight):
    """
    This function should create an instance of a DenseVariational layer according
    to the above specification.
    The function takes the prior_fn, posterior_fn and kl_weight as arguments, which should
    be used to define the layer.
    Your function should then return the layer instance.
    """
    return tfpl.DenseVariational(
        units=10, make_posterior_fn=posterior_fn, make_prior_fn=prior_fn, kl_weight=kl_weight
    )

# Main model functions
tf.random.set_seed(0)
divergence_fn = lambda q, p, _ : tfd.kl_divergence(q, p) / x_train.shape[0]

def get_convolutional_reparameterization_layer(input_shape, divergence_fn):
    """
    This function should create an instance of a Convolution2DReparameterization
    layer according to the above specification.
    The function takes the input_shape and divergence_fn as arguments, which should
    be used to define the layer.
    Your function should then return the layer instance.
    """

    layer = tfpl.Convolution2DReparameterization(
                input_shape=input_shape, filters=8, kernel_size=(5, 5),
                activation='relu', padding='VALID',
                kernel_prior_fn=tfpl.default_multivariate_normal_fn,
                kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                kernel_divergence_fn=divergence_fn,
                bias_prior_fn=tfpl.default_multivariate_normal_fn,
                bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
                bias_divergence_fn=divergence_fn
            )
    return layer
def get_deterministic_model(input_shape, loss, optimizer, metrics):
    """
    This function should build and compile a CNN model according to the above specification.
    The function takes input_shape, loss, optimizer and metrics as arguments, which should be
    used to define and compile the model.
    Your function should return the compiled model.
    """
    model = Sequential([
        Conv2D(kernel_size=(5, 5), filters=8, activation='relu', padding='VALID', input_shape=input_shape),
        MaxPooling2D(pool_size=(6, 6)),
        Flatten(),
        Dense(units=10, activation='softmax')
    ])

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

# Load data
(x_train, y_train, y_train_oh), (x_test, y_test, y_test_oh) = load_data()

# Define layers
convolutional_reparameterization_layer = get_convolutional_reparameterization_layer(
    input_shape=(28, 28, 1), divergence_fn=divergence_fn
)
dense_variational_layer = get_dense_variational_layer(
    get_prior, get_posterior, kl_weight=1/x_train.shape[0]
)

# Create and compile model
bayesian_model = Sequential([
    convolutional_reparameterization_layer,
    MaxPooling2D(pool_size=(6, 6)),
    Flatten(),
    dense_variational_layer,
    tfpl.OneHotCategorical(10, convert_to_tensor_fn=tfd.Distribution.mode)
])
bayesian_model.compile(loss=nll,
              optimizer=RMSprop(),
              metrics=['accuracy'],
              experimental_run_tf_function=False)

# Federated learning simulation function

import numpy as np
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import random

def test_learning_rates(model_fn, data, learning_rates, epochs=10):
    """
    Test multiple learning rates and return the best one based on minimum loss.

    :param model_fn: Function to create a new model instance.
    :param data: Tuple of (x, y) for training data.
    :param learning_rates: List of learning rates to test.
    :param epochs: Number of epochs to train for each learning rate.
    :return: Best learning rate.
    """
    best_lr = learning_rates[0]
    min_loss = float('inf')

    for lr in learning_rates:
        model = model_fn(lr)
        history = model.fit(*data, epochs=epochs, verbose=0)
        avg_loss = np.mean(history.history['loss'])
        # print("Learning rate: ", lr, " Loss = ", avg_loss)

        if avg_loss < min_loss:
            min_loss = avg_loss
            best_lr = lr

    return best_lr

def federated_learning_simulation(client_data, model_fn, rounds=1):
    """
    Simulate federated learning with a fixed learning rate for all clients.

    :param client_data: List of tuples (x, y) representing data for each client
    :param model_fn: Function that returns a new instance of the model
    :param rounds: Number of federated learning rounds
    :return: Trained model
    """
    global_model = model_fn()  # Initialize with a fixed learning rate

    for round_num in range(rounds):
        local_models = []
        metrics = {'loss': [], 'accuracy': []}

        for client_index, (x, y) in enumerate(client_data):
            # Use the fixed learning rate for each client
            local_model = model_fn()
            local_model.set_weights(global_model.get_weights())

            # Train on local data
            history = local_model.fit(x, y, epochs=50, verbose=0)
            local_models.append(local_model)

            # Collect metrics
            metrics['loss'].append(history.history['loss'][0])
            metrics['accuracy'].append(history.history['accuracy'][0])

        # Aggregate the weights of the local models
        new_weights = [np.mean([model.get_weights()[i] for model in local_models], axis=0)
                       for i in range(len(global_model.get_weights()))]
        global_model.set_weights(new_weights)

        # Average and print metrics
        avg_loss = np.mean(metrics['loss'])
        avg_accuracy = np.mean(metrics['accuracy'])
        print(f"Round {round_num + 1}/{rounds}: Average Loss: {avg_loss}, Average Accuracy: {avg_accuracy}")

    return global_model

# Modified function to create a new model instance with a fixed learning rate
def create_model(learning_rate=0.01):
    return get_deterministic_model(
        input_shape=(28, 28, 1),
        loss=SparseCategoricalCrossentropy(),
        optimizer=RMSprop(learning_rate=learning_rate),
        metrics=['accuracy']
    )

num_clients = 5
client_data = []

# Define the proportions for each client
proportions = [0.10, 0.10, 0.15, 0.25, 0.40]  # Total must be 1.0

# Calculate the number of data points for each proportion
total_data_points = len(x_train)
data_sizes = [int(total_data_points * prop) for prop in proportions]

# Adjust the last size to ensure the sum equals total_data_points
data_sizes[-1] = total_data_points - sum(data_sizes[:-1])

start_idx = 0
for size in data_sizes:
    end_idx = start_idx + size
    client_x = x_train[start_idx:end_idx]
    client_y = y_train[start_idx:end_idx]
    client_data.append((client_x, client_y))
    start_idx = end_idx


# Usage in federated learning simulation
federated_model = federated_learning_simulation(client_data, create_model, rounds=10)
