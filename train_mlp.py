import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

from model import build_model

# --- Enable GPU ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # checking for GPU if present yes otherwise no ...otherwise it will use CPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("[CPU] GPU not found, using CPU instead.")

                    # data set file nd defining traning parameters
                     
DATA_PATH = 'keypoint_data.csv' 
MODEL_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'sign_model_mlp_saved')
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-4

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Load and Prepare Data
    try:
        data = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Data file '{DATA_PATH}' not found. Run data_preprocessor.py first.")
        return

    # Separate features (X: 126 columns) and labels (y: last column)
    X = data.iloc[:, :-1].values
    y_raw = data.iloc[:, -1].values
    
    # Encode class labels
    le = LabelEncoder()
    y_int = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    y = tf.keras.utils.to_categorical(y_int, num_classes=n_classes)
    
    # Save the class names for inference
    with open(os.path.join(MODEL_DIR, "classes.txt"), "w") as f:
        for cls in le.classes_:
            f.write(f"{cls}\n")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_int)
    
    # 2. Build and Compile Model
    model = build_model(input_shape=(X_train.shape[1],), n_classes=n_classes)
    
    model.compile(optimizer=Adam(LEARNING_RATE), loss="categorical_crossentropy", metrics=["accuracy"])

    # 3. Training Callbacks
    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    ]

    # 4. Train the Model
    print("\nStarting MLP Model Training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val), 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        callbacks=callbacks
    )

    print(f"\nTraining complete! Best model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()