# DATASET GENERATION

import os
import random
from faker import Faker
from PIL import Image, ImageDraw, ImageFont

# Define directories
output_dir = "small_text_dataset"
images_dir = os.path.join(output_dir, "images")
labels_dir = os.path.join(output_dir, "labels")

# Create directories
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Initialize Faker for generating random text
fake = Faker()

# Define image parameters
IMG_WIDTH, IMG_HEIGHT = 300, 300
NUM_IMAGES = 10  # Small dataset with 10 images
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
    "C:/Windows/Fonts/arial.ttf",  # Windows
    "/Library/Fonts/Arial.ttf"  # Mac
]
FONT_SIZE = 18

# Try to load the font, fallback to default if unavailable
def get_font():
    for path in FONT_PATHS:
        if os.path.exists(path):
            return ImageFont.truetype(path, FONT_SIZE)
    print("Font not found, using default font.")
    return ImageFont.load_default()

# Load the font
font = get_font()

# Generate dataset
for i in range(NUM_IMAGES):
    # Create blank image with random background color
    img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), color=(random.randint(220, 255), random.randint(220, 255), random.randint(220, 255)))
    draw = ImageDraw.Draw(img)

    # Generate random text
    text = fake.word()

    # Get text size using textbbox()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Randomly position the text
    x_min = random.randint(10, IMG_WIDTH - text_width - 10)
    y_min = random.randint(10, IMG_HEIGHT - text_height - 10)
    x_max = x_min + text_width
    y_max = y_min + text_height

    # Draw text on image
    draw.text((x_min, y_min), text, font=font, fill=(0, 0, 0))

    # Save image
    image_filename = f"text_img_{i + 1}.jpg"
    img.save(os.path.join(images_dir, image_filename))

    # Save YOLO annotation (class_id, x_center, y_center, width, height normalized)
    x_center = (x_min + x_max) / 2 / IMG_WIDTH
    y_center = (y_min + y_max) / 2 / IMG_HEIGHT
    box_width = (x_max - x_min) / IMG_WIDTH
    box_height = (y_max - y_min) / IMG_HEIGHT

    label_filename = f"text_img_{i + 1}.txt"
    with open(os.path.join(labels_dir, label_filename), "w") as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

print(f"Small dataset generated successfully! Images and labels saved in '{output_dir}'")
