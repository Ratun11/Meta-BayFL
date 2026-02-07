import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
import numpy as np
import matplotlib.pyplot as plt

tfd = tfp.distributions
tfpl = tfp.layers

plt.rcParams['figure.figsize'] = (10, 6)

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

def get_dense_variational_layer(prior_fn, posterior_fn, kl_weight, num_classes=10):
    """
    This function creates an instance of a DenseVariational layer according
    to the specification and wraps it in a DistributionLambda to output
    a categorical distribution for classification.
    """

    dense_variational_layer = tfpl.DenseVariational(
        units=tfpl.OneHotCategorical.params_size(num_classes),
        make_posterior_fn=posterior_fn,
        make_prior_fn=prior_fn,
        kl_weight=kl_weight,
        activation=None  # No activation, raw logits are needed
    )

    # Wrap the DenseVariational layer in a DistributionLambda layer
    distribution_layer = tfpl.DistributionLambda(
        lambda t: tfd.OneHotCategorical(logits=t),
        name='output_distribution'
    )

    return lambda x: distribution_layer(dense_variational_layer(x))

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
def get_bayesian_model(input_shape, loss, optimizer, metrics, kl_weight):
    model = Sequential([
        tfpl.Convolution2DReparameterization(
            input_shape=input_shape, filters=8, kernel_size=(5, 5),
            activation='relu', padding='VALID',
            kernel_prior_fn=tfpl.default_multivariate_normal_fn,
            kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            kernel_divergence_fn=lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(x_train.shape[0], dtype=tf.float32),
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            bias_divergence_fn=lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(x_train.shape[0], dtype=tf.float32)
        ),
        MaxPooling2D(pool_size=(6, 6)),
        Flatten(),
        tfpl.DenseVariational(
            units=tfpl.OneHotCategorical.params_size(10),
            make_posterior_fn=get_posterior,
            make_prior_fn=get_prior,
            kl_weight=kl_weight,
            activation=None  # No activation, raw logits are needed
        ),
        tfpl.DistributionLambda(lambda t: tfd.OneHotCategorical(logits=t))
    ])

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


# Federated learning simulation function
def federated_learning_simulation(client_data, model_fn, x_test, y_test_oh, rounds=1):
    global_model = model_fn()

    for round_num in range(rounds):
        local_models = []
        metrics = {'loss': [], 'accuracy': []}

        for x, y in client_data:
            local_model = model_fn()
            local_model.set_weights(global_model.get_weights())

            history = local_model.fit(x, y, epochs=50, verbose=0)
            local_models.append(local_model)

            metrics['loss'].append(history.history['loss'][-1])
            metrics['accuracy'].append(history.history['accuracy'][-1])

        new_weights = [np.mean([model.get_weights()[i] for model in local_models], axis=0)
                       for i in range(len(global_model.get_weights()))]
        global_model.set_weights(new_weights)

        avg_loss = np.mean(metrics['loss'])
        avg_accuracy = np.mean(metrics['accuracy'])
        # print(f"Round {round_num + 1}/{rounds}: Average Loss: {avg_loss}, Average Accuracy: {avg_accuracy}")

        # Evaluate on test data
        test_metrics = global_model.evaluate(x_test, y_test_oh, verbose=0)
        print(f"Round {round_num + 1}/{rounds}: Test Loss: {test_metrics[0]}, Test Accuracy: {test_metrics[1]}")

    return global_model

# Function to create a new model instance
def create_model():
    return get_bayesian_model(
    input_shape=(28, 28, 1),
    loss=nll,  # Make sure your nll function is compatible
    optimizer=Adam(),
    metrics=['accuracy'],
    kl_weight=1/x_train.shape[0]
)

# Function to load and preprocess data
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = 1 - x_train / 255.0
    x_test = 1 - x_test / 255.0
    y_train_oh = tf.keras.utils.to_categorical(y_train)
    y_test_oh = tf.keras.utils.to_categorical(y_test)

    # Splitting data for federated learning
    num_clients = 3
    num_samples = len(x_train)
    client1_size = int(num_samples * 0.20)  # 20% of data
    client2_size = int(num_samples * 0.30)  # 30% of data
    # The rest for client 3

    client_indices = np.arange(num_samples)
    # np.random.shuffle(client_indices)  # Shuffle indices

    client_train_dataset = [x_train[:client1_size],
                            x_train[client1_size:client1_size + client2_size],
                            x_train[client1_size + client2_size:]]

    client_train_labels = [y_train_oh[:client1_size],
                           y_train_oh[client1_size:client1_size + client2_size],
                           y_train_oh[client1_size + client2_size:]]

    client_data = [(client_train_dataset[i], client_train_labels[i]) for i in range(num_clients)]

    return client_data, (x_train, x_test, y_test, y_test_oh)

# Load and split the data
client_data, (x_train, x_test, y_test, y_test_oh) = load_data()


# Now that x_train is defined, we can define divergence_fn and dense_variational_layer
tf.random.set_seed(0)
divergence_fn = lambda q, p, _ : tfd.kl_divergence(q, p) / x_train.shape[0]
convolutional_reparameterization_layer = get_convolutional_reparameterization_layer(
    input_shape=(28, 28, 1), divergence_fn=divergence_fn
)
dense_variational_layer =  get_dense_variational_layer(get_prior, get_posterior, kl_weight=1/x_train.shape[0], num_classes=10)

# Use this function to create your Bayesian model
bayesian_model = get_bayesian_model(
    input_shape=(28, 28, 1),
    loss=nll,  # Make sure your nll function is compatible
    optimizer=Adam(),
    metrics=['accuracy'],
    kl_weight=1/x_train.shape[0]
)
bayesian_model.compile(loss=nll,
              optimizer=Adam(),
              metrics=['accuracy'],
              experimental_run_tf_function=False)

# bayesian_model.summary()
# bayesian_model.fit(x=x_train, y=y_train_oh, epochs=50, verbose=True)

# Perform federated learning
federated_model = federated_learning_simulation(client_data, create_model, x_test, y_test_oh, rounds=50)