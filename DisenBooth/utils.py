
from PIL import Image
from PIL import ImageDraw,ImageFont

def render_caption(image, text, coords, font_path='../fonts/GoNotoCurrent.ttf'):
    # Load the font
    draw = ImageDraw.Draw(image)
    font_size = 30  # Starting with the given font size
    font = ImageFont.truetype(font_path, font_size)
    x0, y0, x1, y1 = coords
    # Initial positions
    current_x = x0
    current_y = y0
    line_height = draw.textsize("A", font=font)[1]
    words = text.split()
    for word in words:
        # Calculate the size of the word
        word_width, word_height = draw.textsize(word, font=font)

        # Move to the next line if the word exceeds the width
        if current_x + word_width > x1:
            current_x = x0
            current_y += line_height

        # Check if the text exceeds the height of the bounding box
        if current_y + word_height > y1:
            print("Text does not fit within the given region.")
            return image
        # Draw the word
        draw.text((current_x, current_y), word, font=font, fill="black")
        current_x += word_width + draw.textsize(" ", font=font)[0]  # Add space width

    return image

def float_to_str(f):
    s = f"{f:.15f}"  # Start with a high precision
    return s.rstrip('0').rstrip('.') if '.' in s else s