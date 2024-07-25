import PIL
from PIL import Image
from PIL import ImageFont,ImageDraw
# def render_caption(image,text,coords,font_path='fonts/GoNotoCurrent.ttf'):
#     # Load the font
#     draw = ImageDraw.Draw(image)
#     font_size = 50  # Starting with the smallest possible font size
#     font = ImageFont.truetype(font_path, font_size)
#     x0,y0,x1,y1=coords
#     font = ImageFont.truetype(font_path, font_size)
#     draw.text((x0, y0), text, font=font, fill="black")
#     return image
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
def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def exif_transpose(image):
    """
    If an image has an EXIF Orientation tag, other than 1, return a new image
    that is transposed accordingly. The new image will have the orientation
    data removed.

    Otherwise, return a copy of the image.

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112)
    method = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }.get(orientation)
    if method is not None:
        transposed_image = image.transpose(method)
        transposed_exif = transposed_image.getexif()
        if 0x0112 in transposed_exif:
            del transposed_exif[0x0112]
            if "exif" in transposed_image.info:
                transposed_image.info["exif"] = transposed_exif.tobytes()
            elif "Raw profile type exif" in transposed_image.info:
                transposed_image.info[
                    "Raw profile type exif"
                ] = transposed_exif.tobytes().hex()
            elif "XML:com.adobe.xmp" in transposed_image.info:
                for pattern in (
                    r'tiff:Orientation="([0-9])"',
                    r"<tiff:Orientation>([0-9])</tiff:Orientation>",
                ):
                    transposed_image.info["XML:com.adobe.xmp"] = re.sub(
                        pattern, "", transposed_image.info["XML:com.adobe.xmp"]
                    )
        return transposed_image
    return image.copy()