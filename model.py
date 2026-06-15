import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(126,), n_classes=36):
    """
    Builds a Multi-Layer Perceptron (MLP) for keypoint-based classification.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model