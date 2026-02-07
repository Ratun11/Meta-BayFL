import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop, SGD, Adam

import numpy as np
import os
import matplotlib.pyplot as plt

tfd = tfp.distributions
tfpl = tfp.layers

plt.rcParams['figure.figsize'] = (10, 6)

# print("Tensorflow Version: ", tf.__version__)
# print("Tensorflow Probability Version: ", tfp.__version__)

def inspect_images(data, num_images):
    fig, ax = plt.subplots(nrows=1, ncols=num_images, figsize=(2 * num_images, 2))
    for i in range(num_images):
        # Reshape if necessary
        image = data[i].reshape(28, 28)
        ax[i].imshow(image, cmap='gray')
        ax[i].axis('off')
    plt.show()

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
tf.random.set_seed(0)
deterministic_model = get_deterministic_model(
    input_shape=(28, 28, 1),
    loss=CategoricalCrossentropy(),
    optimizer=Adam(),
    metrics=['accuracy']
)

# deterministic_model.summary()
# deterministic_model.fit(x_train, y_train, epochs=50)

'''
x_plot = np.linspace(-5, 5, 1000)[:, np.newaxis]
plt.plot(x_plot, tfd.Normal(loc=0, scale=1).prob(x_plot).numpy(), label='unit normal', linestyle='--')
plt.plot(x_plot, spike_and_slab(1, dtype=tf.float32).prob(x_plot).numpy(), label='spike and slab')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()
'''
# Federated Learning Simulation Function
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

        # avg_loss = np.mean(metrics['loss'])
        # avg_accuracy = np.mean(metrics['accuracy'])
        # print(f"Round {round_num + 1}/{rounds}: Average Loss: {avg_loss}, Average Accuracy: {avg_accuracy}")

        # Evaluate on test data
        test_metrics = global_model.evaluate(x_test, y_test_oh, verbose=0)
        print(f"Round {round_num + 1}/{rounds}: Test Loss: {test_metrics[0]}, Test Accuracy: {test_metrics[1]}")

    return global_model

# Function to create a new model instance
def create_model():
    return get_deterministic_model(
        input_shape=(28, 28, 1),
        loss=CategoricalCrossentropy(),
        optimizer=Adam(),
        metrics=['accuracy']
    )

# Function to load and preprocess data
def load_data_corrupted():
    # Load original MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize and invert colors
    x_train = 1 - x_train / 255.0
    x_test = 1 - x_test / 255.0

    # Simulate corruption by adding noise
    noise_factor = 0.925
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy,  0., 1.)

    # One-hot encode labels
    y_train_oh = tf.keras.utils.to_categorical(y_train)
    y_test_oh = tf.keras.utils.to_categorical(y_test)

    # Splitting data for federated learning as before
    num_clients = 3
    num_samples = len(x_train_noisy)
    client1_size = int(num_samples * 0.20)
    client2_size = int(num_samples * 0.30)
    # The rest for client 3
    client_indices = np.arange(num_samples)
    np.random.shuffle(client_indices)

    client_train_dataset = [x_train_noisy[:client1_size],
                            x_train_noisy[client1_size:client1_size + client2_size],
                            x_train_noisy[client1_size + client2_size:]]

    client_train_labels = [y_train_oh[:client1_size],
                           y_train_oh[client1_size:client1_size + client2_size],
                           y_train_oh[client1_size + client2_size:]]

    client_data = [(client_train_dataset[i], client_train_labels[i]) for i in range(num_clients)]

    return client_data, (x_train_noisy, x_test_noisy, y_test, y_test_oh)

# Load and split the data
client_data, (x_train, x_test, y_test, y_test_oh) = load_data_corrupted()

# bayesian_model.summary()
# bayesian_model.fit(x=x_train, y=y_train_oh, epochs=50, verbose=True)

# Perform federated learning
federated_model = federated_learning_simulation(client_data, create_model, x_test, y_test_oh, rounds=50)
