import json
import base64

import os
import requests


def submit_post(url: str, data: dict):
    """
    Submit a POST request to the given URL with the given data.
    """
    return requests.post(url, data=json.dumps(data))


def save_encoded_image(b64_image: str, output_path: str):
    """
    Save the given image to the given output path.
    """
    with open(output_path, "wb") as image_file:
        image_file.write(base64.b64decode(b64_image))


def text_query(data: dict, filename: str):
    txt2img_url = 'http://127.0.0.1:7861/sdapi/v1/txt2img'
    response = submit_post(txt2img_url, data)
    output_dir = "disruptor/static/images"
    output_filename = os.path.join(output_dir, filename)
    save_encoded_image(response.json()['images'][0], output_filename)

def image_query(data: dict, filename: str):
    img2img_url = 'http://127.0.0.1:7861/sdapi/v1/img2img'
    response = submit_post(img2img_url, data)
    output_dir = "disruptor\static\images"
    output_filename = os.path.join(output_dir, filename)
    print(response.json())
    save_encoded_image(response.json()['images'][0], output_filename)

def set_deliberate():
    data = {"sd_model_checkpoint": "deliberate_v2.safetensors"}
    options_url = 'http://127.0.0.1:7861/sdapi/v1/options'
    response = submit_post(options_url, data)
    print(response)