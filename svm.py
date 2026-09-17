import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# set working files and directories
train_csv_file = 'train.csv'
test_csv_file  = 'test.csv'
train_folder = 'train'
test_folder  = 'test'

def extract_image_vectors(df, image_dir, include_labels=True):
    # transform images into a list of flattened vectors
     X, Y = [], []

     for idx in range(len(df)):
        path = os.path.join(image_dir, df['id'].iloc[idx])
        img  = Image.open(path).convert('RGB')
        arr  = np.asarray(img, dtype=np.float32).ravel()
        X.append(arr)

     X = np.stack(X)
     if include_labels:
        Y = df['label'].values
        return X, Y
     return X, None

# read and split the dataset
print("Files loading..")
initial_df = pd.read_csv(train_csv_file)
train_df, val_df = train_test_split(initial_df, test_size=0.1, random_state=42, stratify=initial_df['label'])
print(f"Train: {len(train_df)} images | Validation: {len(val_df)} images")

train_features, train_targets = extract_image_vectors(train_df, train_folder, include_labels=True)
val_features,   val_targets   = extract_image_vectors(val_df, train_folder, include_labels=True)

test_df = pd.read_csv(test_csv_file)
test_features, _ = extract_image_vectors(test_df, test_folder, include_labels=False)
print(f"Train samples: {train_features.shape[0]} | Val samples: {val_features.shape[0]} | Test samples: {test_features.shape[0]}")


pipeline = Pipeline([
    ('scaler', StandardScaler()), # normalize data
    ('pca', PCA(n_components=200, random_state=42)), # reduce dimensions to 200 principal components
    ('svc', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)) # SVM model - RBF kernel
])

# train the model
print("Train SVM...")
pipeline.fit(train_features, train_targets)

# evaluate on validation data
print("Evaluating on validation set...")
y_val_pred   = pipeline.predict(val_features)
val_accuracy = accuracy_score(val_targets, y_val_pred)
print(f"Validation accuracy (SVM + PCA): {val_accuracy:.4f}")

# save final predictions
print("Generating predictions on test set...")
y_test_pred = pipeline.predict(test_features)
submission  = pd.DataFrame({
    'id'   : test_df['id'],
    'label': y_test_pred
})
submission.to_csv('sample_submission.csv', index=False)
print("sample_submission.csv has been successfully created!")