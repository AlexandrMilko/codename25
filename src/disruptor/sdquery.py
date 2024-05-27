import json
import base64
from flask import url_for

from disruptor import app
# from disruptor.staging_ml import Room
from disruptor.preprocess_for_empty_space import parse_objects, unite_groups, unite_masks
from flask_login import current_user

import os
import requests

import cv2
from PIL import Image
import math
import shutil

from disruptor.tools import create_directory_if_not_exists, min_max_scale, move_file, submit_post, save_encoded_image, get_encoded_image, run_preprocessor, restart_stable_diffusion

MAX_CONTROLNET_IMAGE_SIZE_KB = 10
MAX_CONTROLNET_IMAGE_RESOLUTION = 600


class Query:
    negative_prompt = "ugly, poorly designed, amateur, bad proportions, bad lighting, direct sunlight, people, person, cartoonish, text"
    sampler_name = "DPM2"


class TextQuery(Query):
    """
        Used for generating favourites for the first time,
        after the user chose the options they like
    """
    steps = 20

    def __init__(self, text, output_filename):
        space, room, budget, style = text.split(", ")
        # We will use it to determine the model
        self.style = style

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
        output_dir = f"disruptor/static/images/{current_user.id}"
        output_filepath = os.path.join(output_dir, self.output_filename)

        # If there was no such dir, we create it and try again
        try:
            save_encoded_image(response.json()['images'][0], output_filepath)
        except FileNotFoundError as e:
            create_directory_if_not_exists(output_dir)
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

        space, room, budget, self.style = text.split(", ")
        self.prompt = f"{room}, {space} space, {budget} budget, {self.style} style, elegant, neat, clean, ultra-realistic, global illumination, unreal engine 5, octane render, highly detailed"
        # self.prompt = text + ", elegant, neat, clean, ultra-realistic, global illumination, unreal engine 5, octane render, highly detailed"
        self.output_filename = output_filename
        self.denoising_strength = denoising_strength

    def run(self):
        if self.style in ("Modern", "Art Deco"):
            set_xsarchitectural()
        else:
            set_deliberate()
        data = {
            'prompt': self.prompt,
            "sampler_name": self.sampler_name,
            "init_images": self.init_images,
            "cfg_scale": self.cfg_scale,
            "denoising_strength": self.denoising_strength,
            "steps": self.steps
        }
        img2img_url = 'http://127.0.0.1:7861/sdapi/v1/img2img'
        response = submit_post(img2img_url, data)
        output_dir = f"disruptor/static/images/{current_user.id}"
        output_filepath = os.path.join(output_dir, self.output_filename)

        # If there was no such dir, we create it and try again
        try:
            save_encoded_image(response.json()['images'][0], output_filepath)
        except FileNotFoundError as e:
            create_directory_if_not_exists(output_dir)
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
        result_path = "disruptor" + url_for('static', filename=f'images/{current_user.id}/{result_filename}')
        self.result_image_b64 = get_encoded_image(result_path)

        # This one will represent the space
        user_path = "disruptor" + url_for('static', filename=f'images/{current_user.id}/{user_filename}')
        # if os.path.getsize(user_path) > MAX_CONTROLNET_IMAGE_SIZE_KB * 1024:
        #     change_image_size(user_path, user_path, MAX_CONTROLNET_IMAGE_SIZE_KB)
        self.user_image_b64 = get_encoded_image(user_path)
        self.width, self.height = get_max_possible_size(user_path)
        # self.set_image_size_from_user_image(user_path)

        space, room, budget, self.style = text.split(", ")
        self.prompt = f'interior design, equipped {room.lower()}, {self.style.lower()} style, ultra-realistic, global illumination, unreal engine 5, octane render, highly detailed, two tone lighting, <lora:epi_noiseoffset2:1>'
        self.output_filename = output_filename

    def run(self):
        if self.style in ("Modern", "Art Deco"):
            set_xsarchitectural()
        else:
            set_deliberate()
        data = {
            "prompt": self.prompt,
            "sampler_name": self.sampler_name,
            # "negative_prompt": self.negative_prompt,
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
                            "image": self.result_image_b64,
                            "module": "seg_ofade20k",
                            "model": "control_sd15_seg [fef5e48e]",
                            "weight": 0.9,
                            "guidance_start": 0.1,
                            "guidance_end": 0.5,
                            # "control_mode": 1,
                            "processor_res": 512
                        },
                        {
                            "image": self.user_image_b64,
                            "module": "softedge_hed",
                            "model": "control_sd15_hed [fef5e48e]",
                            "weight": 0.55,
                            "guidance_start": 0.1,
                            "guidance_end": 0.5,
                            # "control_mode": 0,
                            "processor_res": 512
                        },
                        {
                            "image": self.user_image_b64,
                            "module": "seg_ofade20k",
                            "model": "control_sd15_seg [fef5e48e]",
                            "weight": 0.9,
                            "guidance_start": 0,
                            "guidance_end": 0.5,
                            # "control_mode": 1,
                            "processor_res": 512
                        },
                        {
                            "image": self.user_image_b64,
                            "module": "depth_midas",
                            "model": "control_sd15_depth [fef5e48e]",
                            "weight": 0.4,
                            "guidance_start": 0.1,
                            "guidance_end": 0.5,
                            # "control_mode": 0,
                            "processor_res": 512
                        }
                    ]
                }
            }
        }

        img2img_url = 'http://127.0.0.1:7861/sdapi/v1/img2img'
        response = submit_post(img2img_url, data)
        output_dir = f"disruptor/static/images/{current_user.id}"
        output_filepath = os.path.join(output_dir, self.output_filename)

        # If there was no such dir, we create it and try again
        try:
            save_encoded_image(response.json()['images'][0], output_filepath)
        except FileNotFoundError as e:
            create_directory_if_not_exists(output_dir)
            save_encoded_image(response.json()['images'][0], output_filepath)

    def set_image_size_from_user_image(self, image_path):
        image = cv2.imread(image_path)
        height, width, channels = image.shape
        self.height = height
        self.width = width


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

    def __init__(self, text, output_filename="applied.jpg", prerequisite="prerequisite.png",
                 inpainting_mask="inpainting_mask.png"):
        # We will use result image to transform it into new space of user image
        self.prerequisite_path = "disruptor" + url_for('static',
                                                       filename=f'images/{current_user.id}/preprocessed/{prerequisite}')
        self.inpainting_mask_path = "disruptor" + url_for('static',
                                                          filename=f'images/{current_user.id}/preprocessed/{inpainting_mask}')
        self.prerequisite_image_b64 = get_encoded_image(self.prerequisite_path)
        self.inpainting_mask_image_b64 = get_encoded_image(self.inpainting_mask_path)
        self.width, self.height = get_max_possible_size(self.prerequisite_path)

        space, room, budget, self.style = text.split(", ")
        # self.prompt = f'interior design, {room.lower()}, {self.style.lower()} style, ultra-realistic, global illumination, unreal engine 5, octane render, highly detailed, two tone lighting, <lora:epi_noiseoffset2:1>'
        self.prompt = f'interior design, {room.lower()}, {self.style.lower()} style, RAW photo, subject, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3, <lora:epi_noiseoffset2:1>'
        self.output_filename = output_filename

    def run(self):
        # We run segmentation for our prerequisite image to see if segmentation was done correctly
        run_preprocessor("seg_ofade20k", self.prerequisite_path, current_user.id, "seg_prerequisite.jpg")

        # if self.style in ("Modern", "Art Deco"):
        #     set_xsarchitectural()
        # else:
        #     # set_realistic_vision()
        #     set_deliberate()
        set_realistic_vision()

        self.staged_image_b64 = self.stage()
        self.design()

    def stage(self):
        self.denoising_strength = 1
        self.steps = 20
        data = {
            # "prompt": self.prompt,
            "prompt": "",
            "sampler_name": self.sampler_name,
            # "negative_prompt": self.negative_prompt,
            "init_images": [self.prerequisite_image_b64],
            "batch_size": 1,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "denoising_strength": self.denoising_strength,
            "width": self.width,
            "height": self.height,
            # "seed": 123, # TODO add seed, before testing
            # "mask": self.inpainting_mask_image_b64,
            # "mask_blur": 3,
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "advanced_weighting": None,
                            "animatediff_batch": False,
                            "batch_image_files": [],
                            "batch_images": "",
                            "batch_mask_dir": None,
                            "batch_modifiers": [],
                            "effective_region_mask": None,
                            "hr_option": "Both",
                            "inpaint_crop_input_image": False,
                            "input_mode": "simple",
                            "ipadapter_input": None,
                            "is_ui": False,
                            "loopback": False,
                            "low_vram": False,
                            "mask": None,
                            "output_dir": "",
                            "pixel_perfect": False,
                            "pulid_mode": "Fidelity",
                            "resize_mode": "Crop and Resize",
                            "save_detected_map": True,
                            "threshold_a": 0.5,
                            "threshold_b": 0.5,

                            "enabled": True,
                            "image": self.prerequisite_image_b64,
                            "module": "seg_ofade20k",
                            "model": "control_seg-fp16 [b9c1cc12]",
                            # "low_vram": True,
                            "weight": 1.0,
                            "guidance_start": 0,
                            "guidance_end": 1,
                            "control_mode": "Balanced",
                            "processor_res": 512 # WARNING: TODO change to image height
                        }
                    ]
                }
            }
        }

        img2img_url = 'http://127.0.0.1:7861/sdapi/v1/img2img'
        response = submit_post(img2img_url, data)
        output_dir = f"disruptor/static/images/{current_user.id}/preprocessed"
        output_filepath = os.path.join(output_dir, 'staged.jpg')

        # If there was no such dir, we create it and try again
        try:
            save_encoded_image(response.json()['images'][0], output_filepath)
        except FileNotFoundError as e:
            create_directory_if_not_exists(output_dir)
            save_encoded_image(response.json()['images'][0], output_filepath)

        return response.json()['images'][0]

    def design(self):
        self.denoising_strength = 0.6
        self.cfg_scale = 7
        self.steps = 20

        data = {
            "prompt": self.prompt,
            "sampler_name": self.sampler_name,
            # "negative_prompt": self.negative_prompt,
            "init_images": [self.staged_image_b64],
            "batch_size": 1,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "denoising_strength": self.denoising_strength,
            "width": self.width * 2,
            "height": self.height * 2,
            # "seed": 123, # TODO add seed, before testing
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "advanced_weighting": None,
                            "animatediff_batch": False,
                            "batch_image_files": [],
                            "batch_images": "",
                            "batch_mask_dir": None,
                            "batch_modifiers": [],
                            "effective_region_mask": None,
                            "hr_option": "Both",
                            "inpaint_crop_input_image": False,
                            "input_mode": "simple",
                            "ipadapter_input": None,
                            "is_ui": False,
                            "loopback": False,
                            "low_vram": False,
                            "mask": None,
                            "output_dir": "",
                            "pixel_perfect": False,
                            "pulid_mode": "Fidelity",
                            "resize_mode": "Crop and Resize",
                            "save_detected_map": True,
                            "threshold_a": 0.5,
                            "threshold_b": 0.5,

                            "enabled": True,
                            "image": self.staged_image_b64,
                            "module": "seg_ofade20k",
                            "model": "control_seg-fp16 [b9c1cc12]",
                            "weight": 0.9,
                            "guidance_start": 0.1,
                            "guidance_end": 0.5,
                            "control_mode": "Balanced",
                            "processor_res": 512, # WARNING: TODO change to image height
                            # "low_vram": True,
                        },
                        # {
                        #     "enabled": True,
                        #     "image": self.staged_image_b64,
                        #     "module": "softedge_hed",
                        #     "model": "control_sd15_hed [fef5e48e]",
                        #     "weight": 0.55,
                        #     "guidance_start": 0.1,
                        #     "guidance_end": 0.5,
                        #     # "control_mode": 0,
                        #     "processor_res": 512
                        # },
                        # {
                        #     "enabled": True,
                        #     "image": self.staged_image_b64,
                        #     "module": "seg_ofade20k",
                        #     "model": "control_seg-fp16 [b9c1cc12]",
                        #     "weight": 0.9,
                        #     "guidance_start": 0,
                        #     "guidance_end": 0.5,
                        #     # "control_mode": 1,
                        #     "processor_res": 512,
                        # },
                        {
                            "advanced_weighting": None,
                            "animatediff_batch": False,
                            "batch_image_files": [],
                            "batch_images": "",
                            "batch_mask_dir": None,
                            "batch_modifiers": [],
                            "effective_region_mask": None,
                            "hr_option": "Both",
                            "inpaint_crop_input_image": False,
                            "input_mode": "simple",
                            "ipadapter_input": None,
                            "is_ui": False,
                            "loopback": False,
                            "low_vram": False,
                            "mask": None,
                            "output_dir": "",
                            "pixel_perfect": False,
                            "pulid_mode": "Fidelity",
                            "resize_mode": "Crop and Resize",
                            "save_detected_map": True,
                            "threshold_a": 0.5,
                            "threshold_b": 0.5,

                            "enabled": True,
                            "image": self.staged_image_b64,
                            "module": "depth_midas",
                            "model": "control_depth-fp16 [400750f6]",
                            "weight": 0.4,
                            "guidance_start": 0.1,
                            "guidance_end": 0.5,
                            "control_mode": "Balanced",
                            "processor_res": 512, # WARNING: TODO change to image height
                            # "low_vram": True,
                        }
                    ]
                }
            }
        }
        # data = {
        #     "prompt": self.prompt,
        #     "sampler_name": self.sampler_name,
        #     # "negative_prompt": self.negative_prompt,
        #     "init_images": [self.staged_image_b64],
        #     "batch_size": 1,
        #     "steps": self.steps,
        #     "cfg_scale": self.cfg_scale,
        #     "denoising_strength": self.denoising_strength,
        #     "width": self.width,
        #     "height": self.height,
        #     # "seed": 123, # TODO add seed, before testing
        #     "alwayson_scripts": {
        #         "controlnet": {
        #             "args": [
        #                 {
        #                     "image": self.staged_image_b64,
        #                     "module": "seg_ofade20k",
        #                     "model": "control_seg-fp16 [b9c1cc12]",
        #                     "weight": 1,
        #                     "guidance_start": 0,
        #                     "guidance_end": 1,
        #                     "control_mode": 0,
        #                     "processor_res": 512
        #                 },
        #                 {
        #                     "image": self.staged_image_b64,
        #                     "module": "depth_midas",
        #                     "model": "control_depth-fp16 [400750f6]",
        #                     "weight": 1,
        #                     "guidance_start": 0,
        #                     "guidance_end": 1,
        #                     "control_mode": 0,
        #                     "processor_res": 512
        #                 }
        #             ]
        #         }
        #     }
        # }

        img2img_url = 'http://127.0.0.1:7861/sdapi/v1/img2img'
        response = submit_post(img2img_url, data)
        output_dir = f"disruptor/static/images/{current_user.id}"
        output_filepath = os.path.join(output_dir, self.output_filename)

        # If there was no such dir, we create it and try again
        try:
            save_encoded_image(response.json()['images'][0], output_filepath)
        except FileNotFoundError as e:
            create_directory_if_not_exists(output_dir)
            save_encoded_image(response.json()['images'][0], output_filepath)

    def set_image_size_from_user_image(self, image_path):
        image = cv2.imread(image_path)
        height, width, channels = image.shape
        self.height = height
        self.width = width


# def submit_post(url: str, data: dict):
#     """
#     Submit a POST request to the given URL with the given data.
#     """
#     return requests.post(url, data=json.dumps(data))


# def save_encoded_image(b64_image: str, output_path: str):
#     """
#     Save the given image to the given output path.
#     """
#     with open(output_path, "wb") as image_file:
#         image_file.write(base64.b64decode(b64_image))


def set_deliberate():
    print("SET DELIBERATE")
    data = {"sd_model_checkpoint": "deliberate_v2.safetensors"}
    options_url = 'http://127.0.0.1:7861/sdapi/v1/options'
    response = submit_post(options_url, data)


def set_realistic_vision():
    print("SET realisticVisionV60B1_v51VAE")
    data = {"sd_model_checkpoint": "realisticVisionV60B1_v51VAE.safetensors"}
    options_url = 'http://127.0.0.1:7861/sdapi/v1/options'
    response = submit_post(options_url, data)


def set_xsarchitectural():
    print("SET xsarchitectural")
    data = {"sd_model_checkpoint": "xsarchitectural_v11.ckpt"}
    options_url = 'http://127.0.0.1:7861/sdapi/v1/options'
    response = submit_post(options_url, data)


# def get_encoded_image(image_path):
#     img = cv2.imread(image_path)
#     # Encode into PNG and send to ControlNet
#     retval, bytes = cv2.imencode('.png', img)
#     return base64.b64encode(bytes).decode('utf-8')


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


# def run_preprocessor(preprocessor_name, image_path, filename="preprocessed.jpg", res=512):
#     input_image = get_encoded_image(image_path)
#     data = {
#         "controlnet_module": preprocessor_name,
#         "controlnet_input_images": [input_image],
#         "controlnet_processor_res": res,
#         "controlnet_threshold_a": 64,
#         "controlnet_threshold_b": 64
#     }
#     preprocessor_url = 'http://127.0.0.1:7861/controlnet/detect'
#     response = submit_post(preprocessor_url, data)
#     output_dir = f"disruptor/static/images/{current_user.id}/preprocessed"
#     output_filepath = os.path.join(output_dir, filename)
#
#     # If there was no such dir, we create it and try again
#     try:
#         save_encoded_image(response.json()['images'][0], output_filepath)
#     except FileNotFoundError as e:
#         create_directory_if_not_exists(output_dir)
#         save_encoded_image(response.json()['images'][0], output_filepath)


def prep_bg_image(empty_space):
    """
        Moving our empty space image to background image folder in GracoNet
    """
    image = "disruptor" + url_for('static', filename=f'images/{current_user.id}/{empty_space}')
    shutil.copyfile(image, "../GracoNet-Object-Placement/new_OPA/background/sheep/186413.jpg")


def prep_fg_image(furniture_piece):
    """
        Moving our furniture picture and its map to foreground image folder in GracoNet
    """
    map = "disruptor" + url_for('static', filename=f'images/{current_user.id}/parsed_furniture/{furniture_piece}')
    image = "disruptor" + url_for('static', filename=f'images/{current_user.id}/current_image.jpg')

    shutil.copyfile(image, "../GracoNet-Object-Placement/new_OPA/foreground/sheep/64754.jpg")
    shutil.copyfile(map, "../GracoNet-Object-Placement/new_OPA/foreground/sheep/mask_64754.jpg")


def run_graconet():
    """
        Run graconet to place furniture
    """
    cwd = os.getcwd()
    os.chdir("disruptor")
    os.system("run_graconet_windows.sh")
    os.chdir(cwd)


def remove_files(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        # List all files in the directory
        files = os.listdir(directory_path)

        # Loop through the files and remove them one by one
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                print(f"{file_path} is not a file and won't be removed.")
    else:
        print(f"The directory {directory_path} does not exist.")


def prepare_masks(current_user):
    directory_path = f"disruptor/static/images/{current_user.id}/parsed_furniture"
    # Remove ceiling, walls, to be left only with the objects
    parts_to_remove = ["ceiling", "floor", "wall", "window", "door", "skyscraper", "road",
                       "painting", "curtain", "rail", "stairway", "shelf", "cabinet", "fireplace", "mirror", ]

    # Check if the directory exists
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        # List all files in the directory
        files = os.listdir(directory_path)

        # Loop through the files and remove them one by one
        for file in files:
            file_path = os.path.join(directory_path, file)
            for part in parts_to_remove:
                if part in file:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
    else:
        print(f"The directory {directory_path} does not exist.")


def apply_style(empty_space, room_choice, style_budget_choice):
    es_path = "disruptor" + url_for('static', filename=f'images/{current_user.id}/{empty_space}')
    if room_choice.lower() == "bedroom":
        from disruptor.stage.Bedroom import Bedroom
        room = Bedroom(es_path)
        room.stage(current_user.id)
    elif room_choice.lower() == "kitchen":
        from disruptor.stage.Kitchen import Kitchen
        room = Kitchen(es_path)
        room.stage(current_user.id)
    elif room_choice.lower() == "living room":
        from disruptor.stage.LivingRoom import LivingRoom
        room = LivingRoom(es_path)
        room.stage(current_user.id)
    else:
        raise Exception(f"Wrong Room Type was specified: {room_choice.lower()}")

    # Add time for Garbage Collector
    import time
    time.sleep(5)

    style, budget = style_budget_choice.split(", ")
    # text = f"Residential space, {room_choice}, {budget}, {style}"
    text = f"Residential space, {budget}, {style}"
    query = GreenScreenImageQuery(text)
    query.run()

    # We restart it to deallocate memory. TODO fix it.
    try:
        time.sleep(3)
        restart_stable_diffusion('http://127.0.0.1:7861')
    except requests.exceptions.ConnectionError:
        print("Stable Diffusion restarting")
# def apply_style(empty_space, text):
#     import os
#     es_path = "disruptor" + url_for('static', filename=f'images/{current_user.id}/{empty_space}')
#     run_preprocessor("seg_ofade20k", es_path)
#     user_empty_room = Room(es_path, True)
#
#     import glob
#     room_directory_name = text.split(", ")[1].lower().replace(" ", "_")
#     dataset_originals = f"disruptor/green_screen/find_similar/dataset/{room_directory_name}/original"
#     dataset_paths = glob.glob(os.path.join(dataset_originals, '*.jpg'))
#     max_score = None
#     i = 0
#     paths_count = len(dataset_paths)
#     for dataset_img in dataset_paths:
#         if "before" in dataset_img.lower():
#             dataset_room = Room(dataset_img, False)
#             score = dataset_room.measure_similarity(user_empty_room)
#             if max_score is None:
#                 max_score = score
#                 max_similar = Room.get_trio(dataset_img)["after"]
#                 print(os.path.basename(max_similar), max_score, str(i) + "/" + str(paths_count))
#             elif score > max_score:
#                 max_score = score
#                 max_similar = Room.get_trio(dataset_img)["after"]
#                 print(os.path.basename(max_similar), max_score, str(i) + "/" + str(paths_count))
#         i += 1
#
#     print("Max similar:", os.path.basename(max_similar), max_score)
#     if not max_score > 15:
#         raise Exception("We do not have a similar design in our dataset.")
#
#     # Parse furniture from the selected staged image
#
#     # Create masks
#     run_preprocessor("seg_ofade20k", max_similar)
#     mask_dir = f"disruptor/static/images/{current_user.id}/parsed_furniture"
#     create_directory_if_not_exists(mask_dir)
#     # We update the directory, to get rid of the rubbish from the previous segmentations
#     remove_files(mask_dir)
#     parse_objects(f'disruptor/static/images/{current_user.id}/preprocessed/preprocessed.jpg', current_user.id)
#     prepare_masks(current_user)
#
#     # Create png foreground
#     from disruptor.green_screen.preprocess.create_pngs import create_fg, overlay
#     create_fg(mask_dir, max_similar, current_user.id)
#     fg_path = "disruptor" + url_for('static', filename=f"images/{current_user.id}/preprocessed/foreground.png")
#     create_directory_if_not_exists(os.path.dirname(fg_path))
#     overlay(es_path, fg_path, current_user.id)
#
#     # Create a mask for inpainting
#     mask_path = f'disruptor/static/images/{current_user.id}/preprocessed/inpainting_mask.png'
#     unite_masks(mask_dir, mask_path)
#     mask_img = Image.open(mask_path)
#     prerequisite_path = "disruptor" + url_for('static',
#                                               filename=f'images/{current_user.id}/preprocessed/prerequisite.jpg')
#     prerequisite_img = Image.open(prerequisite_path)
#     mask_img = mask_img.resize(prerequisite_img.size, Image.Resampling.LANCZOS)
#     mask_img.save(mask_path)
#
#     # Make edges smoother
#     from disruptor.preprocess_for_empty_space import perform_dilation
#     perform_dilation(mask_path, mask_path, 16)
#
#     # Run SD to process it
#     query = GreenScreenImageQuery(text)
#     query.run()

# def apply_style(empty_space, text):
#     import os
#     es_path = "disruptor" + url_for('static', filename=f'images/{current_user.id}/{empty_space}')
#
#     run_preprocessor("seg_ofade20k", es_path)
#     user_empty_room = Room(es_path, True) # Just for testing
#
#     import glob
#     room_directory_name = text.split(", ")[1].lower().replace(" ", "_")
#     dataset_originals = f"disruptor/green_screen/find_similar/dataset/{room_directory_name}/original"
#     dataset_paths = glob.glob(os.path.join(dataset_originals, '*.jpg'))
#     max_score = 0
#     i = 0
#     paths_count = len(dataset_paths)
#     for dataset_img in dataset_paths:
#         dataset_room = Room(dataset_img, False)  # For testing
#         score = dataset_room.measure_similarity(user_empty_room)
#         if score > max_score:
#             max_score = score
#             max_similar = Room.get_trio(dataset_img)["after"]
#             print(os.path.basename(max_similar), max_score, str(i) + "/" + str(paths_count))
#         i += 1
#
#     print("Max similar(COS similarity):", os.path.basename(max_similar), max_score)
#
#     # Parse furniture from the selected staged image
#
#     # Create masks
#     run_preprocessor("seg_ofade20k", max_similar)
#     mask_dir = f"disruptor/static/images/{current_user.id}/parsed_furniture"
#     create_directory_if_not_exists(mask_dir)
#     # We update the directory, to get rid of the rubbish from the previous segmentations
#     remove_files(mask_dir)
#     parse_objects(f'disruptor/static/images/{current_user.id}/preprocessed/preprocessed.jpg', current_user.id)
#     prepare_masks(current_user)
#
#     # Create png foreground
#     from disruptor.green_screen.preprocess.create_pngs import create_fg, overlay
#     create_fg(mask_dir, max_similar, current_user.id)
#     fg_path = "disruptor" + url_for('static', filename=f"images/{current_user.id}/preprocessed/foreground.png")
#     create_directory_if_not_exists(os.path.dirname(fg_path))
#     overlay(es_path, fg_path, current_user.id)
#
#     # Create a mask for inpainting
#     mask_path = f'disruptor/static/images/{current_user.id}/preprocessed/inpainting_mask.png'
#     unite_masks(mask_dir, mask_path)
#     mask_img = Image.open(mask_path)
#     prerequisite_path = "disruptor" + url_for('static', filename=f'images/{current_user.id}/preprocessed/prerequisite.jpg')
#     prerequisite_img = Image.open(prerequisite_path)
#     mask_img = mask_img.resize(prerequisite_img.size, Image.Resampling.LANCZOS)
#     mask_img.save(mask_path)
#
#     # Make edges smoother
#     from disruptor.preprocess_for_empty_space import perform_dilation
#     perform_dilation(mask_path, mask_path, 16)
#
#     # Run SD to process it
#     query = GreenScreenImageQuery(text)
#     query.run()

# def apply_style(empty_space, text):
#
#     # Prepare Input
#     import os
#     es_path = "disruptor" + url_for('static', filename=f'images/{current_user.id}/{empty_space}')
#     es_path_resized = "disruptor" + url_for('static', filename=f'images/{current_user.id}/resized_{empty_space}')
#
#     run_preprocessor("seg_ofade20k", es_path)
#     user_empty_room = Room(es_path, True) # Just for testing
#
#     create_directory_if_not_exists(os.path.dirname(es_path))
#
#     # Find the right empty space image
#     from disruptor.green_screen.find_similar.ssim import compare_iou
#     from disruptor.green_screen.find_similar.ssim import compare_vanishing_point, compare_vanishing_point_by_XiaohuLu
#     import glob
#
#     room_directory_name = text.split(", ")[1].lower().replace(" ", "_")
#     segmented_predefined = f"disruptor/green_screen/find_similar/dataset/{room_directory_name}/es_segmented"
#     segmented_paths = glob.glob(os.path.join(segmented_predefined, '*.jpg'))
#     es_number_to_select = 10
#     max_similar_iou_es = dict()
#
#     i = 0
#     comparison_times = len(segmented_paths)
#     # Compare by Intersection over Union
#     for dataset_image_path in segmented_paths:
#
#         # We have to make the images the same size before comparing
#         # Resize
#         es_image = Image.open(es_path)
#         dataset_image = Image.open(dataset_image_path)
#         target_size = dataset_image.size  # Set your desired width and height
#         resized_image = es_image.resize(target_size,
#                                      Image.Resampling.LANCZOS)  # Use a resampling filter for better quality
#         resized_image.save(es_path_resized)
#
#         # Prepare segmented resized image for comparison
#         run_preprocessor("seg_ofade20k", es_path_resized)
#         segmented_path = "disruptor" + url_for('static',
#                                                filename=f'images/{current_user.id}/preprocessed/preprocessed.jpg')
#
#         iou_similarity = compare_iou(segmented_path, dataset_image_path)
#         i += 1
#         print(dataset_image_path, iou_similarity, str(i) + "/" + str(comparison_times))
#
#         #Add to the top similar
#         if len(max_similar_iou_es) < es_number_to_select:
#             max_similar_iou_es[dataset_image_path] = iou_similarity
#         else:
#             min_key = min(max_similar_iou_es, key=lambda k: max_similar_iou_es[k])
#             if max_similar_iou_es[min_key] < iou_similarity:
#                 max_similar_iou_es[dataset_image_path] = iou_similarity
#                 max_similar_iou_es.pop(min_key)
#
#         es_image.close()
#         dataset_image.close()
#         resized_image.close()
#         #
#         # # Just for testing
#         # if i == 20:
#         #     break
#
#     # Calculate relative iou similarity
#
#     min_val = min(max_similar_iou_es.values())
#     max_val = max(max_similar_iou_es.values())
#     # Perform min-max scaling and map to the range [0, 1]
#     max_similar_iou_es = {key: min_max_scale(value, min_val, max_val) for key, value in max_similar_iou_es.items()}
#
#     print(max_similar_iou_es)
#
#     # We calculate vanishing point for the top
#     max_similar_vp_es = dict()
#
#     max_similar_with_cos = None
#     max_similar_score_with_cos = 0
#
#     for dataset_image_path in max_similar_iou_es.keys():
#         # Resize
#         es_image = Image.open(es_path)
#         dataset_image = Image.open(dataset_image_path)
#         target_size = dataset_image.size  # Set your desired width and height
#         resized_image = es_image.resize(target_size,
#                                         Image.Resampling.LANCZOS)  # Use a resampling filter for better quality
#         resized_image.save(es_path_resized)
#         # Prepare segmented resized image for comparison
#         run_preprocessor("seg_ofade20k", es_path_resized)
#         segmented_path = "disruptor" + url_for('static',
#                                                filename=f'images/{current_user.id}/preprocessed/preprocessed.jpg')
#         vp_distance = None
#         try:
#             # TODO Use Room.get_trio
#             import re
#             # Get the room type directory
#             room_directory = os.path.dirname(os.path.dirname(dataset_image_path))
#             corresponding_original_image = str(re.search(r'\d+', os.path.basename(dataset_image_path)).group()) + "Before.jpg"
#             dataset_corresponding_original_path = os.path.join(room_directory, "original/" + corresponding_original_image)
#             vp_distance = compare_vanishing_point_by_XiaohuLu(es_path_resized, dataset_corresponding_original_path)
#
#             room_obj = Room(dataset_image_path, False)  # For testing
#             cos_score = room_obj.measure_similarity(user_empty_room)
#             if cos_score > max_similar_score_with_cos:
#                 max_similar_score_with_cos = cos_score
#                 max_similar_with_cos = dataset_corresponding_original_path
#
#             print(os.path.basename(es_path_resized), os.path.basename(dataset_corresponding_original_path), vp_distance)
#         except Exception as e:
#             print(e)
#             continue
#         finally:
#             if vp_distance is not None:
#                 max_similar_vp_es[dataset_image_path] = vp_distance
#
#             es_image.close()
#             dataset_image.close()
#             resized_image.close()
#
#     # We standardize values
#     min_val = min(max_similar_vp_es.values())
#     max_val = max(max_similar_vp_es.values())
#     max_similar_vp_es = {key: min_max_scale(value, min_val, max_val) for key, value in max_similar_vp_es.items()}
#
#     # Calculate overall similarity score (the more the better)
#     # score = iou / vp_distance
#     scores = dict()
#     for dataset_image_path in max_similar_vp_es.keys():
#         iou = max_similar_iou_es[dataset_image_path]
#         vp_distance = max_similar_vp_es[dataset_image_path]
#         scores[dataset_image_path] = iou - vp_distance
#
#     best_fit = max(scores, key=lambda k: scores[k])
#     print(scores)
#     #
#     # best_fit = "44"
#     # room_directory_name = "living_room"
#     # Find corresponding staged image
#     import re
#     max_similar_stage = str(re.search(r'\d+', os.path.basename(best_fit)).group()) + "After.jpg"
#     max_similar_stage_path = f"disruptor/green_screen/find_similar/dataset/{room_directory_name}/original/" + max_similar_stage
#     print("Max similar: ", max_similar_stage_path)
#     print("Max similar(COS): ", max_similar_with_cos)
#
#     # Parse furniture from the selected staged image
#
#     # Create masks
#     run_preprocessor("seg_ofade20k", max_similar_stage_path)
#     mask_dir = f"disruptor/static/images/{current_user.id}/parsed_furniture"
#     create_directory_if_not_exists(mask_dir)
#     # We update the directory, to get rid of the rubbish from the previous segmentations
#     remove_files(mask_dir)
#     parse_objects(f'disruptor/static/images/{current_user.id}/preprocessed/preprocessed.jpg', current_user.id)
#     prepare_masks(current_user)
#
#     # Create png foreground
#     from disruptor.green_screen.preprocess.create_pngs import create_fg, overlay
#     create_fg(mask_dir, max_similar_stage_path, current_user.id)
#     fg_path = "disruptor" + url_for('static', filename=f"images/{current_user.id}/preprocessed/foreground.png")
#     create_directory_if_not_exists(os.path.dirname(fg_path))
#     overlay(es_path, fg_path, current_user.id)
#
#     # Create a mask for inpainting
#     mask_path = f'disruptor/static/images/{current_user.id}/preprocessed/inpainting_mask.png'
#     unite_masks(mask_dir, mask_path)
#     mask_img = Image.open(mask_path)
#     prerequisite_path = "disruptor" + url_for('static', filename=f'images/{current_user.id}/preprocessed/prerequisite.jpg')
#     prerequisite_img = Image.open(prerequisite_path)
#     mask_img = mask_img.resize(prerequisite_img.size, Image.Resampling.LANCZOS)
#     mask_img.save(mask_path)
#
#     # Make edges smoother
#     from disruptor.preprocess_for_empty_space import perform_dilation
#     perform_dilation(mask_path, mask_path, 16)
#
#     # Run SD to process it
#     query = GreenScreenImageQuery(text)
#     query.run()
#
#     #TODO close all images after opening
#
#     # style="current_image.jpg"
#     # style_image_path = "disruptor" + url_for('static', filename=f'images/{style}')
#     # run_preprocessor("seg_ofade20k", style_image_path)
#     # parse_objects()
#     #
#     # furniture_dir = "disruptor/static/images/parsed_furniture"
#     # groups_dir = "disruptor/static/images/parsed_furniture"
#     # unite_groups(furniture_dir, groups_dir, [["bed", "blanket;cover", "cushion", "pillow"], ["table", "pot", "plant;flora;plant;life"]])
#     #
#     # prep_bg_image(empty_space)
#     # prep_fg_image("bed_blanket;cover_cushion_pillow.jpg")
#     # run_graconet()
