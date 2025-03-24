import os
import json
import pytesseract
import numpy as np
from PIL import Image

# Define dataset directories
output_dir = "text_dataset"
images_dir = os.path.join(output_dir, "images")
annotations_path = os.path.join(output_dir, "annotations", "annotations.json")

# Load annotations
with open(annotations_path, "r") as f:
    annotations = json.load(f)

# Extract text from images and prepare dataset
examples = []  # Feature set
labels = []    # Target (0 = no text, 1 = text detected)

for img_info in annotations["images"]:
    image_path = os.path.join(images_dir, img_info["file_name"])
    img = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(img).strip()
    
    # Features: (length, contains numbers, contains special chars)
    feature = [
        len(extracted_text),
        any(char.isdigit() for char in extracted_text),
        any(not char.isalnum() and not char.isspace() for char in extracted_text)
    ]
    examples.append(feature)
    labels.append(1 if extracted_text else 0)

# Convert to NumPy arrays
examples = np.array(examples)
labels = np.array(labels)

# Find-S Algorithm
def find_s(examples, labels):
    """
    Implements the Find-S algorithm.
    Finds the most specific hypothesis that fits all positive examples.
    """
    # Initialize with the first positive example
    for i, label in enumerate(labels):
        if label == 1:  # Looking for the first positive example
            specific_h = list(map(str, examples[i]))  # Convert elements to strings
            break
    else:
        raise ValueError("No positive examples found!")

    # Update hypothesis based on other positive examples
    for i, label in enumerate(labels):
        if label == 1:
            example = list(map(str, examples[i]))  # Ensure example values are strings
            for j in range(len(specific_h)):
                if specific_h[j] != example[j]:
                    specific_h[j] = '?'  # Generalize attribute
    return specific_h


# Candidate Elimination Algorithm
def candidate_elimination(examples, labels):
    num_features = len(examples[0])
    S = [None] * num_features  # Most specific hypothesis
    G = [['?'] * num_features]  # Most general hypothesis
    
    for i, example in enumerate(examples):
        if labels[i] == 1:  # Positive Example
            for j in range(num_features):
                if S[j] is None:
                    S[j] = example[j]
                elif S[j] != example[j]:
                    S[j] = '?'  # Generalize S
            
            G = [g for g in G if all(g[j] == '?' or g[j] == S[j] for j in range(num_features))]
        else:  # Negative Example
            new_G = []
            for g in G:
                for j in range(num_features):
                    if g[j] == '?':
                        new_hypothesis = g.copy()
                        new_hypothesis[j] = S[j]
                        new_G.append(new_hypothesis)
            G = new_G
    
    return S, G

# Run Algorithms
specific_hypothesis = find_s(examples, labels)
specific_h, general_h = candidate_elimination(examples, labels)

# Print results
print("Find-S Hypothesis:", specific_hypothesis)
print("Candidate Elimination Hypotheses:")
print("Specific Hypothesis:", specific_h)
print("General Hypotheses:", general_h)
