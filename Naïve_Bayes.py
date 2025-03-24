# trains and tests model using Naïve Bayes

import os
import platform
import pytesseract
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Set Tesseract path dynamically based on OS
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
elif platform.system() == "Darwin":  # macOS
    pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"
else:  # Linux
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Define dataset path
images_dir = "small_text_dataset/images"
if not os.path.exists(images_dir):
    raise FileNotFoundError(f"Error: The directory '{images_dir}' does not exist. Generate the dataset first!")

# Define categories and labels
categories = ["tech", "food", "sports", "fashion"]
NUM_IMAGES = 10  # Number of generated images

# Extracted text and labels
extracted_texts = []
labels = []

# Extract text from generated images and assign labels
for i in range(NUM_IMAGES):
    image_path = os.path.join(images_dir, f"text_img_{i + 1}.jpg")

    if not os.path.exists(image_path):
        print(f"Warning: File '{image_path}' not found. Skipping...")
        continue

    # Extract text using Tesseract
    extracted_text = pytesseract.image_to_string(Image.open(image_path)).strip()

    # Check if extracted text is empty
    if not extracted_text:
        print(f"Warning: No text found in '{image_path}', assigning default text.")
        extracted_text = "default"

    extracted_texts.append(extracted_text)

    # Assign random labels cyclically for simplicity
    label = i % len(categories)  # Rotating category labels
    labels.append(label)

print(" Text extracted from images successfully!")

# Convert text to feature vectors using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(extracted_texts)  # Vectorize extracted text
y = np.array(labels)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎉 Model Accuracy: {accuracy * 100:.2f}%")

#  Test with a new text sample
new_text = "pizza"  # Sample text to classify
new_text_vectorized = vectorizer.transform([new_text])
predicted_label = model.predict(new_text_vectorized)[0]
predicted_category = categories[predicted_label]

print(f"The word '{new_text}' is classified under '{predicted_category}'!")
