import json
import base64
from flask import url_for

import os
import requests

import cv2
from PIL import Image
import math

MAX_CONTROLNET_IMAGE_SIZE_KB = 10
MAX_CONTROLNET_IMAGE_RESOLUTION = 600

class Query:
    negative_prompt = "ugly, poorly designed, amateur, bad proportions, bad lighting, direct sunlight, people, person, cartoonish, text"
    sampler_name = "DPM2 Karras"
class TextQuery(Query):
    """
        Used for generating favourites for the first time,
        after the user chose the options they like
    """
    steps = 20
    def __init__(self, text, output_filename):
        space, room, budget, style = text.split(", ")
        self.style = style
        print(self.style, "STYLE")

        if self.style == "Modern":
            self.prompt = f"{room}, {space} space, {budget} budget, sleek, minimalistic, functional, open, neutral, tech-influenced style, elegant, neat, clean, ultra-realistic, global illumination, unreal engine 5, octane render, highly detailed"
        elif self.style == "Art Deco":
            self.prompt = f"{room}, {space} space, {budget} budget, opulent, glamorous, geometric, luxurious, vintage, ornate style, elegant, neat, clean, ultra-realistic, global illumination, unreal engine 5, octane render, highly detailed"
        else:
            self.prompt = f"{room}, {space} space, {budget} budget, {style} style, elegant, neat, clean, ultra-realistic, global illumination, unreal engine 5, octane render, highly detailed"

        # self.prompt = text + ", elegant, neat, clean, ultra-realistic, global illumination, unreal engine 5, octane render, highly detailed"
        self.output_filename = output_filename

    def run(self):
        if self.style in ("Modern", "Art Deco"):
            set_xsarchitectural()
        else:
            set_deliberate()
        data = {
            'prompt': self.prompt,
            "sampler_name": self.sampler_name,
            "negative_prompt": self.negative_prompt,
            "steps": self.steps
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
    steps = 20
    def __init__(self, text, image_url, output_filename, denoising_strength):
        # Setting up the input image
        image_path = "disruptor" + image_url
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            input_image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        self.init_images = [input_image_b64]

        space, room, budget, style = text.split(", ")
        self.prompt = f"{room}, {space} space, {budget} budget, {style} style, elegant, neat, clean, ultra-realistic, global illumination, unreal engine 5, octane render, highly detailed"
        # self.prompt = text + ", elegant, neat, clean, ultra-realistic, global illumination, unreal engine 5, octane render, highly detailed"
        self.output_filename = output_filename
        self.denoising_strength = denoising_strength

    def run(self):
        set_deliberate()
        data = {
            'prompt': self.prompt,
            "sampler_name": self.sampler_name,
            "init_images": self.init_images,
            "cfg_scale": self.cfg_scale,
            "denoising_strength": self.denoising_strength,
            "negative_prompt": self.denoising_strength,
            "steps": self.steps
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
    steps = 40
    def __init__(self, text, user_filename, output_filename, result_filename="current_image.jpg"):
        # We will use result image to transform it into new space of user image
        result_path = "disruptor" + url_for('static', filename=f'images/{result_filename}')
        self.result_image_b64 = get_encoded_image(result_path)

        # This one will represent the space
        user_path = "disruptor" + url_for('static', filename=f'images/{user_filename}')
        # if os.path.getsize(user_path) > MAX_CONTROLNET_IMAGE_SIZE_KB * 1024:
        #     change_image_size(user_path, user_path, MAX_CONTROLNET_IMAGE_SIZE_KB)
        self.user_image_b64 = get_encoded_image(user_path)
        self.width, self.height = get_max_possible_size(user_path)
        # self.set_image_size_from_user_image(user_path)

        space, room, budget, style = text.split(", ")
        self.prompt = f'interior design, equipped {room.lower()}, {style.lower()} style, ultra-realistic, global illumination, unreal engine 5, octane render, highly detailed, two tone lighting, <lora:epi_noiseoffset2:1>'
        self.output_filename = output_filename

    def run(self):
        # data = {
        #     "prompt": self.prompt,
        #     "sampler_name": self.sampler_name,
        #     "negative_prompt": self.negative_prompt,
        #     "init_images": [self.result_image_b64],
        #     "batch_size": 1,
        #     "steps": self.steps,
        #     "cfg_scale": self.cfg_scale,
        #     "denoising_strength": self.denoising_strength,
        #     "width": self.width,
        #     "height": self.height,
        #     "seed": 123,
        #     "alwayson_scripts": {
        #         "controlnet": {
        #             "args": [
        #                 {
        #                     "input_image": self.result_image_b64,
        #                     "module": "seg_ofade20k",
        #                     "model": "control_sd15_seg [fef5e48e]",
        #                     "weight": 0.9,
        #                     "guidance_start": 0.1,
        #                     "guidance_end": 0.5,
        #                     "control_mode": 0,
        #                     "processor_res": 512
        #                 },
        #                 {
        #                     "input_image": self.user_image_b64,
        #                     "module": "softedge_hed",
        #                     "model": "control_sd15_hed [fef5e48e]",
        #                     "weight": 0.55,
        #                     "guidance_start": 0.1,
        #                     "guidance_end": 0.5,
        #                     "control_mode": 0,
        #                     "processor_res": 512
        #                 },
        #                 {
        #                     "input_image": self.user_image_b64,
        #                     "module": "seg_ofade20k",
        #                     "model": "control_sd15_seg [fef5e48e]",
        #                     "weight": 0.6,
        #                     "guidance_start": 0,
        #                     "guidance_end": 0.5,
        #                     "control_mode": 0,
        #                     "processor_res": 512
        #                 },
        #                 {
        #                     "input_image": self.user_image_b64,
        #                     "module": "depth_midas",
        #                     "model": "control_sd15_depth [fef5e48e]",
        #                     "weight": 0.4,
        #                     "guidance_start": 0.1,
        #                     "guidance_end": 0.5,
        #                     "control_mode": 0,
        #                     "processor_res": 512
        #                 }
        #             ]
        #         }
        #     }
        # }

        set_deliberate()
        data = {
            "prompt": self.prompt,
            "sampler_name": self.sampler_name,
            "negative_prompt": self.negative_prompt,
            "init_images": [self.result_image_b64],
            "batch_size": 1,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "denoising_strength": self.denoising_strength,
            "width": self.width,
            "height": self.height,
            # "seed": 123, # TODO add seed, before testing
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
                            "control_mode": 1,
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
                            "weight": 0.9,
                            "guidance_start": 0,
                            "guidance_end": 0.5,
                            "control_mode": 1,
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

        # Print params
        data.pop("init_images")
        data["alwayson_scripts"]["controlnet"]["args"][0].pop("input_image")
        data["alwayson_scripts"]["controlnet"]["args"][1].pop("input_image")
        data["alwayson_scripts"]["controlnet"]["args"][2].pop("input_image")
        data["alwayson_scripts"]["controlnet"]["args"][3].pop("input_image")
        print(data)

    def set_image_size_from_user_image(self, image_path):
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
    print("SET DELIBERATE")
    data = {"sd_model_checkpoint": "deliberate_v2.safetensors"}
    options_url = 'http://127.0.0.1:7861/sdapi/v1/options'
    response = submit_post(options_url, data)

def set_xsarchitectural():
    print("SET xsarchitectural")
    data = {"sd_model_checkpoint": "xsarchitectural_v11.ckpt"}
    options_url = 'http://127.0.0.1:7861/sdapi/v1/options'
    response = submit_post(options_url, data)

def get_encoded_image(image_path):
    img = cv2.imread(image_path)
    # Encode into PNG and send to ControlNet
    retval, bytes = cv2.imencode('.png', img)
    return base64.b64encode(bytes).decode('utf-8')

def change_image_size(input_path, output_path, target_size_kb=20):
    # Load the image using Pillow
    img = Image.open(input_path)

    # Set the maximum size (20 KB) in bytes
    target_size_bytes = target_size_kb * 1024

    # Calculate the current size of the image in bytes
    img_bytes = os.path.getsize(input_path)

    # Calculate the required compression ratio to achieve the target size
    compression_ratio = math.sqrt(target_size_bytes / img_bytes)

    # Calculate the new dimensions
    new_width = int(img.width * compression_ratio)
    new_height = int(img.height * compression_ratio)

    # Resize the image while preserving the aspect ratio
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Save the resized image to the output path
    resized_img.save(output_path)


def get_max_possible_size(input_path, target_resolution=MAX_CONTROLNET_IMAGE_RESOLUTION):
    # Load the image using Pillow
    img = Image.open(input_path)

    # Get the current dimensions of the image
    width, height = img.size

    # Check if either dimension is greater than target_resolution pixels
    if width > target_resolution or height > target_resolution:
        # Calculate the new dimensions while preserving the aspect ratio
        if width > height:
            new_width = target_resolution
            new_height = int(height * (target_resolution / width))
        else:
            new_width = int(width * (target_resolution / height))
            new_height = target_resolution

        # Resize the image using the calculated dimensions
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return new_width, new_height

    # If no resizing was done, return the original dimensions
    return width, height

def run_preprocessor(preprocessor_name, image_path):
    input_image = get_encoded_image(image_path)
    data = {
        "controlnet_module": preprocessor_name,
        "controlnet_input_images": [input_image],
        "controlnet_processor_res": 512,
        "controlnet_threshold_a": 64,
        "controlnet_threshold_b": 64
    }
    preprocessor_url = 'http://127.0.0.1:7861/controlnet/detect'
    response = submit_post(preprocessor_url, data)
    output_dir = "disruptor/static/images/preprocessed"
    output_filepath = os.path.join(output_dir, "preprocessed.jpg")
    save_encoded_image(response.json()['images'][0], output_filepath)