import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam # NEW: Import Adam optimizer
import os

# --- Configuration ---
DATA_PATH = 'keypoint_data.csv'
MODEL_PATH = 'models/sign_model_mlp_saved'

def main():
    # 1. Load Model and Data
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        data = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f" Error loading model or data: {e}")
        return
        
    # Recompile the model to enable evaluation                 
    model.compile(optimizer=Adam(), 
                  loss="categorical_crossentropy", 
                  metrics=["accuracy"])
    print("Model recompiled successfully for evaluation.")


    # 2. Prepare Data
    X = data.iloc[:, :-1].values #for keypointers 
    y_raw = data.iloc[:, -1].values # for class labels
    
    le = LabelEncoder()
    y_int = le.fit_transform(y_raw)
    y = tf.keras.utils.to_categorical(y_int, num_classes=len(le.classes_))
    
    # Re-split to get the same validation set as in training
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_int)
    y_val_true = np.argmax(y_val, axis=1) # True integer labels

    # 3. Predict and Evaluate
    print("\nEvaluating Model on Validation Set...")
    # This call now works because the model is compiled.
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    
    # model.predict() works even without explicit compilation, but it's executed here.
    y_pred_probs = model.predict(X_val, verbose=0) 
    y_pred_classes = np.argmax(y_pred_probs, axis=1) # Predicted integer labels

    # 4. Report Results
    print(f"\n--- Evaluation Results ---")
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_val_true, y_pred_classes, target_names=le.classes_, zero_division=0))

if __name__ == "__main__":
    main()