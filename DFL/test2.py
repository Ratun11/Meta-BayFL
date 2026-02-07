import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

def create_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(11,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def get_dataset():
    (train_dataset, _), ds_info = tfds.load(
        'wine_quality',
        split=['train', 'test'],
        as_supervised=True,
        with_info=True,
    )

    def preprocess(features, target):
        # Normalize features for better model performance
        return tf.cast(features, tf.float32) / 255.0, tf.cast(target, tf.float32)

    train_dataset = train_dataset.map(preprocess).batch(10)
    return train_dataset


def distribute_datasets(train_dataset, distribution):
    distributed_datasets = []
    iterator = iter(train_dataset.unbatch())
    for count in distribution:
        dataset_slice = tf.data.Dataset.from_tensor_slices(
            [next(iterator) for _ in range(count)]
        ).batch(10)
        distributed_datasets.append(dataset_slice)
    return distributed_datasets


def train_local_models(distributed_datasets):
    local_models = []
    for dataset in distributed_datasets:
        model = create_model()
        model.fit(dataset, epochs=1, verbose=0)
        local_models.append(model)
    return local_models


def federated_averaging(local_models):
    global_model = create_model()
    global_weights = global_model.get_weights()

    for layer in range(len(global_weights)):
        layer_weights = np.array([model.get_weights()[layer] for model in local_models])
        averaged_weights = np.mean(layer_weights, axis=0)
        global_weights[layer] = averaged_weights

    global_model.set_weights(global_weights)
    return global_model


def evaluate_global_model(global_model, evaluation_dataset):
    loss, mse = global_model.evaluate(evaluation_dataset, verbose=0)
    print(f"Global model evaluation - Loss: {loss}, MSE: {mse}")



train_dataset = get_dataset()
evaluation_dataset = train_dataset  # For demonstration, use separate dataset in practice
distributed_datasets = distribute_datasets(train_dataset, [20, 30, 50])
local_models = train_local_models(distributed_datasets)
global_model = federated_averaging(local_models)
evaluate_global_model(global_model, evaluation_dataset)
