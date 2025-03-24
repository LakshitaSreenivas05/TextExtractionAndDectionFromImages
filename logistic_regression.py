#LOGISTIC REGRESSION

import os
import json
import pytesseract
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Define dataset directories
output_dir = "text_dataset"
images_dir = os.path.join(output_dir, "images")
annotations_path = os.path.join(output_dir, "annotations", "annotations.json")

# Load annotations
with open(annotations_path, "r") as f:
    annotations = json.load(f)

# Extract text from images and prepare dataset
texts = []
labels = []  # Binary label: 0 = short text, 1 = long text

# Define length threshold to classify text
LENGTH_THRESHOLD = 50  # Texts longer than 50 characters are considered "long"

for img_info in annotations["images"]:
    image_path = os.path.join(images_dir, img_info["file_name"])

    # Load image
    img = Image.open(image_path)

    # Extract text using pytesseract
    extracted_text = pytesseract.image_to_string(img).strip()

    # Skip empty text
    if extracted_text:
        texts.append(extracted_text)

        # Label based on text length
        if len(extracted_text) > LENGTH_THRESHOLD:
            labels.append(1)  # Long text
        else:
            labels.append(0)  # Short text

# Check if we have enough data
if len(texts) < 10:
    raise ValueError("Not enough valid text data for classification. Try generating more images.")

# Visualize Text Length Distribution
plt.figure(figsize=(8, 5))
sns.histplot([len(text) for text in texts], bins=20, kde=True, color='skyblue')
plt.title("📏 Text Length Distribution")
plt.xlabel("Text Length")
plt.ylabel("Frequency")
plt.show()

# Convert text data to Bag of Words (BoW) using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Create and train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Make predictions
y_pred = lr_model.predict(X_test)
y_prob = lr_model.predict_proba(X_test)[:, 1]

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

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="orange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

# Generate Word Clouds
short_texts = " ".join([text for text, label in zip(texts, labels) if label == 0])
long_texts = " ".join([text for text, label in zip(texts, labels) if label == 1])

# Word Cloud for Short Texts
plt.figure(figsize=(12, 6))
wc_short = WordCloud(width=500, height=300, background_color="white").generate(short_texts)
plt.imshow(wc_short, interpolation="bilinear")
plt.title("Word Cloud for Short Texts")
plt.axis("off")
plt.show()

# Word Cloud for Long Texts
plt.figure(figsize=(12, 6))
wc_long = WordCloud(width=500, height=300, background_color="white").generate(long_texts)
plt.imshow(wc_long, interpolation="bilinear")
plt.title("Word Cloud for Long Texts")
plt.axis("off")
plt.show()

# Test with a new image
def predict_text_class(image_path):
    img = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(img).strip()

    if not extracted_text:
        print("No text detected in the image.")
        return

    # Transform the text to BoW representation
    X_new = vectorizer.transform([extracted_text])
    predicted_class = lr_model.predict(X_new)[0]

    # Class prediction
    if predicted_class == 1:
        print(f"Predicted Class: Long Text (>{LENGTH_THRESHOLD} characters)")
    else:
        print(f"Predicted Class: Short Text (<={LENGTH_THRESHOLD} characters)")

# Example usage:
# predict_text_class("text_dataset/images/text_image_5.jpg")
