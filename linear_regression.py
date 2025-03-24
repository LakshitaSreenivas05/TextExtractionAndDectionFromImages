# LINEAR REGRESSION

import os
import json
import pytesseract
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Define dataset directories
output_dir = "text_dataset"
images_dir = os.path.join(output_dir, "images")
annotations_path = os.path.join(output_dir, "annotations", "annotations.json")

# Load annotations
with open(annotations_path, "r") as f:
    annotations = json.load(f)

# Extract text from images and prepare dataset
texts = []
text_lengths = []  # Length of the extracted text (target for regression)

for img_info in annotations["images"]:
    image_path = os.path.join(images_dir, img_info["file_name"])

    # Load image
    img = Image.open(image_path)

    # Extract text using pytesseract
    extracted_text = pytesseract.image_to_string(img).strip()

    # Skip empty text
    if extracted_text:
        texts.append(extracted_text)
        text_lengths.append(len(extracted_text))  # Target value: text length

# Check if we have enough data
if len(texts) < 10:
    raise ValueError("Not enough valid text data for regression. Try generating more images.")

# Convert text data to Bag of Words (BoW) using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, text_lengths, test_size=0.25, random_state=42)

# Create and train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred = lr_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Test with a new image
def predict_text_length(image_path):
    img = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(img).strip()

    if not extracted_text:
        print("No text detected in the image.")
        return

    # Transform the text to BoW representation
    X_new = vectorizer.transform([extracted_text])
    predicted_length = lr_model.predict(X_new)[0]

    print(f"Predicted Text Length: {int(predicted_length)} characters")

# Example usage:
# predict_text_length("text_dataset/images/text_image_5.jpg")
