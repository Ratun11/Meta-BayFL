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
    input_shape=(32, 32, 3),
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

        avg_loss = np.mean(metrics['loss'])
        avg_accuracy = np.mean(metrics['accuracy'])
        # print(f"Round {round_num + 1}/{rounds}: Average Loss: {avg_loss}, Average Accuracy: {avg_accuracy}")

        # Evaluate on test data
        test_metrics = global_model.evaluate(x_test, y_test_oh, verbose=0)
        print(f"Round {round_num + 1}/{rounds}: Test Loss: {test_metrics[0]}, Test Accuracy: {test_metrics[1]}")

    return global_model

# Function to create a new model instance
def create_model():
    return get_deterministic_model(
        input_shape=(32, 32, 3),
        loss=CategoricalCrossentropy(),
        optimizer=Adam(),
        metrics=['accuracy']
    )

# Function to load and preprocess data
import numpy as np
import tensorflow as tf


# Function to load and preprocess data
# Function to load and preprocess CIFAR-10 data
def load_data_cifar10():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize the images to be between 0 and 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # One-hot encode the labels
    y_train_oh = tf.keras.utils.to_categorical(y_train, 10)
    y_test_oh = tf.keras.utils.to_categorical(y_test, 10)

    # Define the number of clients and calculate the number of samples per client
    num_clients = 3
    num_samples_per_client = [int(len(x_train) * 0.20), int(len(x_train) * 0.30), int(len(x_train) * 0.50)]

    # Shuffle the indices to distribute the data among clients
    client_indices = np.arange(len(x_train))
    np.random.shuffle(client_indices)

    # Split the training data and labels among clients
    client_train_dataset = [x_train[sum(num_samples_per_client[:i]):sum(num_samples_per_client[:i + 1])] for i in
                            range(num_clients)]
    client_train_labels = [y_train_oh[sum(num_samples_per_client[:i]):sum(num_samples_per_client[:i + 1])] for i in
                           range(num_clients)]

    # Package the data into a list of tuples (one for each client)
    client_data = list(zip(client_train_dataset, client_train_labels))

    # Return the client data and the full training and testing datasets with one-hot labels
    return client_data, (x_train, y_train_oh, x_test, y_test_oh)


# Call the function to get the data
client_data, (x_train, y_train_oh, x_test, y_test_oh) = load_data_cifar10()

# bayesian_model.summary()
# bayesian_model.fit(x=x_train, y=y_train_oh, epochs=50, verbose=True)

# Perform federated learning
federated_model = federated_learning_simulation(client_data, create_model, x_test, y_test_oh, rounds=50)
