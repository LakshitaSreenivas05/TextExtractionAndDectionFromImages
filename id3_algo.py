import os
import json
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define dataset directories
output_dir = "text_dataset"
images_dir = os.path.join(output_dir, "images")
annotations_path = os.path.join(output_dir, "annotations", "annotations.json")

# Load annotations
with open(annotations_path, "r") as f:
    annotations = json.load(f)

# Extract text from images and prepare dataset
texts = []
labels = []  # 0 = short text, 1 = long text

# Define threshold to classify text as short or long
LENGTH_THRESHOLD = 50  

for img_info in annotations["images"]:
    image_path = os.path.join(images_dir, img_info["file_name"])

    # Load image
    img = Image.open(image_path)

    # Extract text using pytesseract
    extracted_text = pytesseract.image_to_string(img).strip()

    # Skip empty text
    if extracted_text:
        texts.append(extracted_text)
        labels.append(1 if len(extracted_text) > LENGTH_THRESHOLD else 0)

# Ensure enough data for classification
if len(texts) < 10:
    raise ValueError("Not enough valid text data for classification. Try generating more images.")

# Convert text data to numerical features using Bag of Words (BoW)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Create and train ID3 Decision Tree model
id3_model = DecisionTreeClassifier(criterion="entropy", random_state=42)  
id3_model.fit(X_train, y_train)

# Make predictions
y_pred = id3_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Short", "Long"], yticklabels=["Short", "Long"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Predict on a new image
def predict_text_class(image_path):
    img = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(img).strip()

    if not extracted_text:
        print("No text detected in the image.")
        return

    # Transform text to BoW
    X_new = vectorizer.transform([extracted_text]).toarray()
    predicted_class = id3_model.predict(X_new)[0]

    # Display Prediction
    if predicted_class == 1:
        print(f"Predicted Class: Long Text (>{LENGTH_THRESHOLD} characters)")
    else:
        print(f"Predicted Class: Short Text (≤{LENGTH_THRESHOLD} characters)")

# Example usage:
# predict_text_class("text_dataset/images/sample_image.jpg")
