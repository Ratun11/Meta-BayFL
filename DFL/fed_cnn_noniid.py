import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical

def create_cnn_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        # Output layer without activation to output logits
        Dense(10)  # Removed the softmax activation
    ])
    # Use CategoricalCrossentropy with from_logits=True for NLL loss
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def split_data_non_iid(x, y, distribution):
    assert sum(distribution) == 1, "The sum of distribution percentages must be 1."
    total_size = len(x)
    data_splits = [int(total_size * part) for part in distribution]
    indices = np.arange(total_size)
    np.random.shuffle(indices)
    x_shuffled = x[indices]
    y_shuffled = y[indices]
    data = []
    start = 0
    for split in data_splits:
        end = start + split
        data.append((x_shuffled[start:end], y_shuffled[start:end]))
        start = end
    return data

# Load and preprocess the data
(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()  # Change this to mnist for using MNIST data
# Selecting 10% of the data
indices = np.arange(len(x_train_full))
np.random.shuffle(indices)
reduced_size = int(0.5 * len(indices))
x_train_reduced = x_train_full[indices[:reduced_size]]
y_train_reduced = y_train_full[indices[:reduced_size]]

x_train = x_train_reduced.reshape(x_train_reduced.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train_reduced, 10)
y_test = to_categorical(y_test, 10)

# Proceed as before with your modifications
distribution = [0.20, 0.30, 0.50]  # Adjust distribution if needed
client_data = split_data_non_iid(x_train, y_train, distribution)

# Federated learning simulation with 3 clients
def federated_learning_simulation(client_data, x_test, y_test, global_epochs=100):
    global_model = create_cnn_model()
    for global_epoch in range(global_epochs):
        sum_weights = [np.zeros_like(w) for w in global_model.get_weights()]
        for x_client, y_client in client_data:
            local_model = create_cnn_model()
            local_model.set_weights(global_model.get_weights())
            local_model.fit(x_client, y_client, epochs=1, verbose=0)
            for i, w in enumerate(local_model.get_weights()):
                sum_weights[i] += w
        average_weights = [w / len(client_data) for w in sum_weights]
        global_model.set_weights(average_weights)
        loss, accuracy = global_model.evaluate(x_test, y_test, verbose=0)
        print(f"Global Epoch {global_epoch+1}/{global_epochs}: Test Accuracy: {accuracy}, Test Loss: {loss}")
    return global_model

# Perform federated learning simulation
federated_learning_simulation(client_data, x_test, y_test)
