#DATASET GENERATION

import os
import random
import json
from faker import Faker
from PIL import Image, ImageDraw, ImageFont

# Define directories
output_dir = "text_dataset"
images_dir = os.path.join(output_dir, "images")
annotations_dir = os.path.join(output_dir, "annotations")

# Create directories
os.makedirs(images_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)

# Initialize Faker for generating random text
fake = Faker()

# Define image parameters
IMG_WIDTH, IMG_HEIGHT = 640, 480
NUM_IMAGES = 100  # Number of images to generate
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Change if needed
FONT_SIZE = 20

# Create annotations dictionary for COCO format
annotations = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "text"}]}
annotation_id = 1

for i in range(NUM_IMAGES):
    # Create blank image with random background color
    img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), color=(random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)))
    draw = ImageDraw.Draw(img)

    # Generate random text
    text = fake.sentence(nb_words=random.randint(2, 6))
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    # Get text size using textbbox()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Check if the text fits within the image dimensions
    if text_width >= IMG_WIDTH - 20 or text_height >= IMG_HEIGHT - 20:
        print(f"Skipping text that doesn't fit in image: '{text}'")
        continue

    # Generate random coordinates to place the text
    x_min = random.randint(10, IMG_WIDTH - text_width - 10)
    y_min = random.randint(10, IMG_HEIGHT - text_height - 10)
    x_max = x_min + text_width
    y_max = y_min + text_height

    # Draw text on image
    draw.text((x_min, y_min), text, font=font, fill=(0, 0, 0))

    # Save image
    image_filename = f"text_image_{i + 1}.jpg"
    img.save(os.path.join(images_dir, image_filename))

    # Add image to COCO annotations
    annotations["images"].append({
        "id": i + 1,
        "file_name": image_filename,
        "width": IMG_WIDTH,
        "height": IMG_HEIGHT
    })

    # Add bounding box annotation
    annotations["annotations"].append({
        "id": annotation_id,
        "image_id": i + 1,
        "category_id": 1,
        "bbox": [x_min, y_min, text_width, text_height],
        "area": text_width * text_height,
        "iscrowd": 0
    })
    annotation_id += 1

# Save annotations in COCO format
with open(os.path.join(annotations_dir, "annotations.json"), "w") as f:
    json.dump(annotations, f, indent=4)

print(f"Dataset generated successfully! Images and annotations saved in '{output_dir}'")
