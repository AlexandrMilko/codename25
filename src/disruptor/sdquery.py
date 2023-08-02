import json
import base64
from flask import url_for

import os
import requests

import cv2
from PIL import Image

class Query:
    negative_prompt = "ugly, poorly designed, amateur, bad proportions, bad lighting, direct sunlight, people, person, cartoonish, text"
    sampler_name = "DPM2 Karras"
class TextQuery(Query):
    """
        Used for generating favourites for the first time,
        after the user chose the options they like
    """
    def __init__(self, text, output_filename):
        self.prompt = text + ", elegant, neat, clean, ultra-realistic, global illumination, unreal engine 5, octane render, highly detailed"
        self.output_filename = output_filename

    def run(self):
        data = {
            'prompt': self.prompt,
            "sampler_name": self.sampler_name,
            "negative_prompt": self.negative_prompt
        }
        txt2img_url = 'http://127.0.0.1:7861/sdapi/v1/txt2img'
        response = submit_post(txt2img_url, data)
        output_dir = "disruptor/static/images"
        output_filepath = os.path.join(output_dir, self.output_filename)
        save_encoded_image(response.json()['images'][0], output_filepath)
class ImageQuery(Query):
    """
        Used for generating favourites
        based on the "favourite" image the user had chosen before that
    """
    cfg_scale = 7.5
    def __init__(self, text, image_url, output_filename, denoising_strength):
        # Setting up the input image
        image_path = "disruptor" + image_url
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            input_image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        self.init_images = [input_image_b64]

        self.prompt = text + ", elegant, neat, clean, ultra-realistic, global illumination, unreal engine 5, octane render, highly detailed"
        self.output_filename = output_filename
        self.denoising_strength = denoising_strength

    def run(self):
        data = {
             'prompt': self.prompt,
             "sampler_name": self.sampler_name,
             "init_images": self.init_images,
             "cfg_scale": self.cfg_scale,
             "denoising_strength": self.denoising_strength,
             "negative_prompt": self.denoising_strength,
        }
        img2img_url = 'http://127.0.0.1:7861/sdapi/v1/img2img'
        response = submit_post(img2img_url, data)
        output_dir = "disruptor\static\images"
        output_filepath = os.path.join(output_dir, self.output_filename)
        save_encoded_image(response.json()['images'][0], output_filepath)
class ControlNetImageQuery(Query):
    """
        Used for applying the predefined result image to a new space
        (Premium Feature)
    """
    # TODO:
    # automatic width, height
    denoising_strength = 1
    cfg_scale = 7
    sampling_steps = 40
    def __init__(self, text, user_filename, output_filename, result_filename="current_image.jpg"):
        # We will use result image to transform it into new space of user image
        result_path = "disruptor" + url_for('static', filename=f'images/{result_filename}')
        self.result_image_b64 = get_encoded_image(result_path)

        # This one will represent the space
        user_path = "disruptor" + url_for('static', filename=f'images/{user_filename}')
        self.user_image_b64 = get_encoded_image(user_path)
        self.set_image_size(user_path)

        # self.prompt = f'interior design, equipped, {text}, ultra-realistic, global illumination, unreal engine 5, octane render, highly detailed, two tone lighting, <lora:epi_noiseoffset2:1>'
        self.prompt = "interior design, equipped bedroom, scandinavian style, ultra-realistic, global illumination, unreal engine 5, octane render, highly detailed, two tone lighting, <lora:epi_noiseoffset2:1>"
        self.output_filename = output_filename

    def run(self):
        data = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "init_images": [self.result_image_b64],
            "batch_size": 1,
            "steps": self.sampling_steps,
            "cfg_scale": self.cfg_scale,
            "denoising_strength": self.denoising_strength,
            "width": self.width,
            "height": self.height,
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "input_image": self.result_image_b64,
                            "module": "seg_ofade20k",
                            "model": "control_sd15_seg [fef5e48e]",
                            "weight": 0.9,
                            "guidance_start": 0.1,
                            "guidance_end": 0.5,
                            "control_mode": 0,
                            "processor_res": 512
                        },
                        {
                            "input_image": self.user_image_b64,
                            "module": "softedge_hed",
                            "model": "control_sd15_hed [fef5e48e]",
                            "weight": 0.55,
                            "guidance_start": 0.1,
                            "guidance_end": 0.5,
                            "control_mode": 0,
                            "processor_res": 512
                        },
                        {
                            "input_image": self.user_image_b64,
                            "module": "seg_ofade20k",
                            "model": "control_sd15_seg [fef5e48e]",
                            "weight": 0.6,
                            "guidance_start": 0,
                            "guidance_end": 0.5,
                            "control_mode": 0,
                            "processor_res": 512
                        },
                        {
                            "input_image": self.user_image_b64,
                            "module": "depth_midas",
                            "model": "control_sd15_depth [fef5e48e]",
                            "weight": 0.4,
                            "guidance_start": 0.1,
                            "guidance_end": 0.5,
                            "control_mode": 0,
                            "processor_res": 512
                        }
                    ]
                }
            }
        }

        img2img_url = 'http://127.0.0.1:7861/sdapi/v1/img2img'
        response = submit_post(img2img_url, data)
        output_dir = "disruptor\static\images"
        output_filepath = os.path.join(output_dir, self.output_filename)
        save_encoded_image(response.json()['images'][0], output_filepath)

    def set_image_size(self, image_path):
        image = cv2.imread(image_path)
        height, width, channels = image.shape
        self.height = height
        self.width = width

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

def set_deliberate():
    data = {"sd_model_checkpoint": "deliberate_v2.safetensors"}
    options_url = 'http://127.0.0.1:7861/sdapi/v1/options'
    response = submit_post(options_url, data)
    print(response)

def get_encoded_image(image_path):
    img = cv2.imread(image_path)
    # Encode into PNG and send to ControlNet
    retval, bytes = cv2.imencode('.png', img)
    return base64.b64encode(bytes).decode('utf-8')