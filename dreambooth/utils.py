
from PIL import Image
from PIL import ImageDraw,ImageFont
def invert_scientific_notation(value):
    value=str(value).replace('e-0','e-')
    if 'e' in value:
        base, exponent = value.split('e')
        # base = float(base)
        exponent = int(exponent)
        
        # Invert the sign of the exponent
        inverted_exponent = -exponent
        
        # Calculate the new base by adjusting the exponent
        # new_base = base * (10 ** abs(inverted_exponent))
        
        # Convert the base to integer and return in scientific notation format
        return f"{(base)}e{inverted_exponent}"

    else:
        raise ValueError("The input is not in a valid scientific notation format.")
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

def format_exponent(value):
    # Convert the float to a string in scientific notation
    sci_str = "{:e}".format(value)
    # Replace 'e-' with 'e_' for negative exponents
    if 'e-' in sci_str:
        sci_str = sci_str.replace('e-', 'e_')
    # Replace 'e+' with 'e_' for positive exponents
    elif 'e+' in sci_str:
        sci_str = sci_str.replace('e+', 'e_')
    return sci_str