import os
import requests

from run import SD_DOMAIN
import cv2
from PIL import Image
import math

from tools import submit_post, save_encoded_image, get_encoded_image, get_image_size
from constants import Path

MAX_CONTROLNET_IMAGE_SIZE_KB = 10
MAX_CONTROLNET_IMAGE_RESOLUTION = 512


class Query:
    negative_prompt = "ugly, poorly designed, amateur, bad proportions, bad lighting, direct sunlight, people, person, cartoonish, text"
    sampler_name = "DPM2"


class GreenScreenImageQuery(Query):
    """
        Used for applying the predefined result image to a new space
        (Premium Feature)
    """
    # TODO:
    # automatic width, height
    denoising_strength = 1
    cfg_scale = 7
    steps = 20

    def __init__(self, text):
        # We will use result image to transform it into new space of user image
        self.prerequisite_path = Path.RENDER_IMAGE.value
        self.prerequisite_image_b64 = get_encoded_image(self.prerequisite_path)
        # self.width, self.height = get_max_possible_size(self.prerequisite_path)
        self.width, self.height = get_image_size(self.prerequisite_path)

        space, room, budget, self.style = text.split(", ")
        # self.prompt = f'interior design, {room.lower()}, {self.style.lower()} style, ultra-realistic, global illumination, unreal engine 5, octane render, highly detailed, two tone lighting, <lora:epi_noiseoffset2:1>'
        self.prompt = f'{self.style.lower()} style, high-end budget, RAW photo, subject, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3, <lora:epi_noiseoffset2:1>'
        # self.prompt = f'{self.style.lower()} {room}'

    def run(self):
        # if self.style in ("Modern", "Art Deco"):
        #     set_xsarchitectural()
        # else:
        #     # set_realistic_vision()
        #     set_deliberate()
        set_realistic_vision()

        self.designed_image_b64 = self.design()
        self.add_shadows_and_light()

    def design(self):
        self.denoising_strength = 0.75
        self.cfg_scale = 7
        self.steps = 20

        data = {
            "prompt": self.prompt,
            # "prompt": "",
            "sampler_name": self.sampler_name,
            # "negative_prompt": self.negative_prompt,
            # "init_images": [self.prerequisite_image_b64],
            "batch_size": 1,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            # "denoising_strength": self.denoising_strength,
            "width": self.width,
            "height": self.height,
            "seed": -1,
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "enabled": True,
                            "image": self.prerequisite_image_b64,
                            "module": "seg_ofade20k",
                            "model": "control_v11p_sd15_seg [e1f51eb9]",
                            # "low_vram": True,
                            "weight": 1.0,
                            "guidance_start": 0,
                            "guidance_end": 1,
                            "control_mode": "ControlNet is more important",
                            "processor_res": 512  # WARNING: TODO change to image height
                        }
                    ]
                }
            }
        }

        img2img_url = f'http://{SD_DOMAIN}:7861/sdapi/v1/txt2img'
        response = submit_post(img2img_url, data)
        output_filepath = Path.DESIGNED_IMAGE.value

        # If there was no such dir, we create it and try again
        try:
            save_encoded_image(response.json()['images'][0], output_filepath)
        except FileNotFoundError as e:
            print(e)

        return response.json()['images'][0]

    def add_shadows_and_light(self):
        # self.denoising_strength = 0.6
        # self.steps = 20
        # data = {
        #     "prompt": self.prompt,
        #     # "prompt": "",
        #     "sampler_name": self.sampler_name,
        #     # "negative_prompt": self.negative_prompt,
        #     "init_images": [self.designed_image_b64],
        #     "batch_size": 1,
        #     "steps": self.steps,
        #     "cfg_scale": self.cfg_scale,
        #     "denoising_strength": self.denoising_strength,
        #     "width": self.width * 2,
        #     "height": self.height * 2,
        #     "seed": -1,
        #     "alwayson_scripts": {
        #         "controlnet": {
        #             "args": [
        #                 {
        #                     "enabled": True,
        #                     "image": self.prerequisite_image_b64,
        #                     "module": "seg_ofade20k",
        #                     "model": "control_v11p_sd15_seg [e1f51eb9]",
        #                     # "low_vram": True,
        #                     "weight": 1,
        #                     "guidance_start": 0,
        #                     "guidance_end": 1,
        #                     "control_mode": "ControlNet is more important",
        #                     "processor_res": 512 # WARNING: TODO change to image height
        #                 },
        #                 # {
        #                 #     "enabled": True,
        #                 #     "image": self.prerequisite_image_b64,
        #                 #     "module": "depth_anything",
        #                 #     "model": "control_v11f1p_sd15_depth [cfd03158]",
        #                 #     "weight": 0.4,
        #                 #     "guidance_start": 0.1,
        #                 #     "guidance_end": 0.5,
        #                 #     "control_mode": "My prompt is more important",
        #                 #     "processor_res": 512,  # WARNING: TODO change to image height
        #                 #     # "low_vram": True,
        #                 # }
        #             ]
        #         }
        #     }
        # }
        #
        # img2img_url = 'http://127.0.0.1:7861/sdapi/v1/img2img'
        # response = submit_post(img2img_url, data)
        output_filepath = Path.OUTPUT_IMAGE.value
        #
        # # If there was no such dir, we create it and try again
        # try:
        #     save_encoded_image(response.json()['images'][0], output_filepath)
        # except FileNotFoundError as e:
        #     create_directory_if_not_exists(output_dir)
        #     save_encoded_image(response.json()['images'][0], output_filepath)
        save_encoded_image(self.designed_image_b64, output_filepath)

    def set_image_size_from_user_image(self, image_path):
        image = cv2.imread(image_path)
        height, width, channels = image.shape
        self.height = height
        self.width = width


def set_deliberate():
    print("SET DELIBERATE")
    data = {"sd_model_checkpoint": "deliberate_v2.safetensors"}
    options_url = f'http://{SD_DOMAIN}:7861/sdapi/v1/options'
    response = submit_post(options_url, data)


def set_realistic_vision():
    print("SET realisticVisionV60B1_v51VAE")
    data = {"sd_model_checkpoint": "realisticVisionV60B1_v51HyperVAE.safetensors"}
    options_url = f'http://{SD_DOMAIN}:7861/sdapi/v1/options'
    response = submit_post(options_url, data)


def set_xsarchitectural():
    print("SET xsarchitectural")
    data = {"sd_model_checkpoint": "xsarchitectural_v11.ckpt"}
    options_url = f'http://{SD_DOMAIN}:7861/sdapi/v1/options'
    response = submit_post(options_url, data)


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