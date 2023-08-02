import json
import base64

import os
import requests

class Query:
    negative_prompt = "ugly, poorly designed, amateur, bad proportions, bad lighting, direct sunlight, people, person, cartoonish, text"
    sampler_name = "DPM2 Karras"
class TextQuery(Query):
    """
        Used for generating favourites for the first time,
        after the user chose the options they like
    """
    def __init__(self, text, result_filename):
        self.prompt = text + ", elegant, neat, clean, ultra-realistic, global illumination, unreal engine 5, octane render, highly detailed"
        self.result_filename = result_filename

    def run(self):
        data = {
            'prompt': self.prompt,
            "sampler_name": self.sampler_name,
            "negative_prompt": self.negative_prompt
        }
        txt2img_url = 'http://127.0.0.1:7861/sdapi/v1/txt2img'
        response = submit_post(txt2img_url, data)
        result_dir = "disruptor/static/images"
        result_filepath = os.path.join(result_dir, self.result_filename)
        save_encoded_image(response.json()['images'][0], result_filepath)
class ImageQuery(Query):
    """
        Used for generating favourites
        based on the "favourite" image the user had chosen before that
    """
    cfg_scale = 7.5
    def __init__(self, text, image_url, result_filename, denoising_strength):
        # Setting up the input image
        image_path = "disruptor" + image_url
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            input_image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        self.init_images = [input_image_b64]

        self.prompt = text + ", elegant, neat, clean, ultra-realistic, global illumination, unreal engine 5, octane render, highly detailed"
        self.result_filename = result_filename
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
        result_dir = "disruptor\static\images"
        result_filepath = os.path.join(result_dir, self.result_filename)
        save_encoded_image(response.json()['images'][0], result_filepath)
class ControlNetImageQuery(Query):
    """
        Used for applying the predefined result image to a new space
        (Premium Feature)
    """
    # TODO:
    # init_images
    # prompt
    # negative_prompt
    # sampling_steps
    # automatic width, height
    # controlnet_models
    denoising_strength = 1
    cfg_scale = 7
    def __int__(self, text, image_url, result_filename):
        # Setting up the input image
        image_path = "disruptor" + image_url
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            input_image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        self.init_images = [input_image_b64]

        self.prompt = f'interior design, equipped, {text}, ultra-realistic, global illumination, unreal engine 5, octane render, highly detailed, two tone lighting, <lora:epi_noiseoffset2:1>'
        self.result_filename = result_filename
        self.sampling_steps = 40

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