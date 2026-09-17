import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

# standardize the data
print("Normalizam caracteristicile...")
feature_scaler = StandardScaler()
train_features_scaled = feature_scaler.fit_transform(train_features)
val_features_scaled   = feature_scaler.transform(val_features)
test_features_scaled  = feature_scaler.transform(test_features)

# extract principal components
# using 200 components to keep the essential image information
print("Reducing dimensions with PCA...")
pca_reducer = PCA(n_components=200, random_state=42)
train_pca = pca_reducer.fit_transform(train_features_scaled)
val_pca = pca_reducer.transform(val_features_scaled)
test_pca = pca_reducer.transform(test_features_scaled)

explained_variance = pca_reducer.explained_variance_ratio_.sum()
print(f"Variance captured by PCA model: {explained_variance:.2%}")

# define and train the model
print("Initializing and training the Naive Bayes model...")
gaussian_nb_classifier = GaussianNB()
gaussian_nb_classifier.fit(train_pca, train_targets)


# evaluate on validation data
print("Calculating accuracy...")
val_predictions = gaussian_nb_classifier.predict(val_pca)
accuracy_score_val = accuracy_score(val_targets, val_predictions)
print(f"Recorded accuracy (GNB + PCA): {accuracy_score_val:.4f}")

# save final predictions
print("Generating submission file...")
test_predictions = gaussian_nb_classifier.predict(test_pca)
submission_df = pd.DataFrame({
    'id'   : test_df['id'],
    'label': test_predictions
})
submission_df.to_csv('sample_submission.csv', index=False)
print("sample_submission.csv has been successfully created!")