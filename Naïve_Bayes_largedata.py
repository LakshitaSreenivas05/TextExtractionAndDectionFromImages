# Naïve Bayes

import os
import json
import pytesseract
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Define dataset directories
output_dir = "text_dataset"
images_dir = os.path.join(output_dir, "images")
annotations_path = os.path.join(output_dir, "annotations", "annotations.json")

# Load annotations
with open(annotations_path, "r") as f:
    annotations = json.load(f)

# Extract text from images and prepare dataset
texts = []
labels = []  # We'll classify based on text length (just for demo purposes)

for img_info in annotations["images"]:
    image_path = os.path.join(images_dir, img_info["file_name"])

    # Load image
    img = Image.open(image_path)

    # Extract text using pytesseract
    extracted_text = pytesseract.image_to_string(img).strip()

    # Skip empty text
    if extracted_text:
        texts.append(extracted_text)

        # Dummy labels based on text length (for simplicity)
        if len(extracted_text) < 50:
            labels.append(0)  # Short text
        else:
            labels.append(1)  # Long text

# Check if we have enough data
if len(texts) < 10:
    raise ValueError("Not enough valid text data for classification. Try generating more images.")

# Convert text data to Bag of Words (BoW) using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Create and train Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Make predictions
y_pred = nb_classifier.predict(X_test)

# Evaluate the model
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Test with a new image
def predict_text_category(image_path):
    img = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(img).strip()
    if not extracted_text:
        print("No text detected in the image.")
        return

    # Transform the text to BoW representation
    X_new = vectorizer.transform([extracted_text])
    prediction = nb_classifier.predict(X_new)[0]

    if prediction == 0:
        print(f"Prediction: Short text detected.")
    else:
        print(f"Prediction: Long text detected.")

# Example usage:
# predict_text_category("text_dataset/images/text_image_5.jpg")
