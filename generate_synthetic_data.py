import pandas as pd
import numpy as np
import os

def main():
    print("Generating synthetic keypoint dataset...")
    
    # Define classes: A-Z and 0-9 (36 classes)
    classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + [str(i) for i in range(10)]
    n_classes = len(classes)
    n_samples_per_class = 100
    n_features = 126
    
    # Generate reproducible class centers
    np.random.seed(42)
    class_centers = np.random.uniform(0.1, 0.9, size=(n_classes, n_features))
    
    data_rows = []
    
    for class_idx, class_name in enumerate(classes):
        center = class_centers[class_idx]
        for _ in range(n_samples_per_class):
            # Add small noise to make it realistic but highly learnable
            noise = np.random.normal(0, 0.05, size=(n_features,))
            features = np.clip(center + noise, 0.0, 1.0)
            row = list(features) + [class_name]
            data_rows.append(row)
            
    columns = [f'feature_{i}' for i in range(n_features)] + ['class']
    df = pd.DataFrame(data_rows, columns=columns)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    DATA_PATH = 'keypoint_data.csv'
    df.to_csv(DATA_PATH, index=False)
    print(f"Generated {len(df)} synthetic samples across {n_classes} classes and saved to '{DATA_PATH}'")

if __name__ == '__main__':
    main()
