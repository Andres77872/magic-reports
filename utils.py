import base64
import requests
from PIL import Image
from io import BytesIO

def scale_image(image, new_height=1536):
    aspect_ratio = image.width / image.height
    new_width = int(new_height * aspect_ratio)
    return image.resize((new_width, new_height))

def fetch_and_encode_image(url, new_height=1024):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        resized_image = scale_image(image, new_height)

        buffered = BytesIO()
        resized_image.save(buffered, format="PNG")
        encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return encoded
    except Exception as e:
        print(f'Image fetch/processing error: {e}')
        return None