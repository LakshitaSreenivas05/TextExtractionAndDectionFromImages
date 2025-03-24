from PIL import Image, ImageDraw, ImageFont

# Use a system font that exists
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SIZE = 20

# Load font correctly
try:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
except OSError:
    print("⚠️ Font not found, using default font.")
    font = ImageFont.load_default()
