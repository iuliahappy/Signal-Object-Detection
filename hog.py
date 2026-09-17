import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# feature Extraction Function
def process_images_to_array(image_names, folder_name):
    feature_matrix = []
    
    for index, filename in enumerate(image_names):
        full_path = os.path.join(folder_name, filename)
        
        # using grayscale
        gray_image = Image.open(full_path).convert('L')
        pixel_array = np.array(gray_image)
        
        # extract hog descriptors
        descriptor = hog( pixel_array, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
        
        feature_matrix.append(descriptor)
        
        if (index + 1) % 1000 == 0 or (index + 1) == len(image_names):
            print(f"Processed {index + 1}/{len(image_names)} images...")
            
    return np.array(feature_matrix)


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print("Splitting data for local validation...")
train_split, validation_split = train_test_split(train_data, test_size=0.15, stratify=train_data['label'], random_state=42)

print("Extracting HOG arrays...")
print("Processing training images...")
x_train = process_images_to_array(train_split['id'].values, 'train')
y_train = train_split['label'].values

print("Processing validation images...")
x_val = process_images_to_array(validation_split['id'].values, 'train')
y_val = validation_split['label'].values

print("Processing test images...")
x_test = process_images_to_array(test_data['id'].values, 'test')

# define and train the model
print("\nTraining the Random Forest model...")
model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
model.fit(x_train, y_train)

print("Evaluating on validation set...")
predictions_val = model.predict(x_val)
accuracy = accuracy_score(y_val, predictions_val) * 100
print(f"Validation Accuracy: {accuracy:.2f}%\n")

print("Generating final test predictions...")
predictions_test = model.predict(x_test)

# save final predictions
print("Saving submission file...")
submission_df = pd.DataFrame({
    'id': test_data['id'],
    'label': predictions_test
})
submission_df.to_csv('sample_submission_HOGG.csv', index=False)
        
print("sample_submission has been successfully created!")