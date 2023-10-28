import json
import base64
from flask import url_for

from disruptor import app
from disruptor.preprocess_for_empty_space import parse_objects, unite_groups
from flask_login import current_user

import os
import requests

import cv2
from PIL import Image
import math
import shutil

from disruptor.tools import create_directory_if_not_exists

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
            "negative_prompt": self.denoising_strength,
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
    steps = 40
    def __init__(self, text, output_filename="applied.jpg", prerequisite="prerequisite.jpg"):
        # We will use result image to transform it into new space of user image
        prerequisite_path = "disruptor" + url_for('static', filename=f'images/{current_user.id}/preprocessed/{prerequisite}')
        self.prerequisite_image_b64 = get_encoded_image(prerequisite_path)
        self.width, self.height = get_max_possible_size(prerequisite_path)

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
            "negative_prompt": self.negative_prompt,
            "init_images": [self.prerequisite_image_b64],
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
                            "input_image": self.prerequisite_image_b64,
                            "module": "seg_ofade20k",
                            "model": "control_sd15_seg [fef5e48e]",
                            "weight": 1,
                            "guidance_start": 0,
                            "guidance_end": 1,
                            "control_mode": 0,
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
    print(os.getcwd())
    print(image_path)
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
    output_dir = f"disruptor/static/images/{current_user.id}/preprocessed"
    output_filepath = os.path.join(output_dir, "preprocessed.jpg")

    # If there was no such dir, we create it and try again
    try:
        save_encoded_image(response.json()['images'][0], output_filepath)
    except FileNotFoundError as e:
        create_directory_if_not_exists(output_dir)
        save_encoded_image(response.json()['images'][0], output_filepath)

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
    parts_to_remove = ["ceiling", "floor", "wall", "window", "door", "skyscraper", "road"]

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

def apply_style(empty_space, text):

    # Prepare Input
    import os
    es_path = "disruptor" + url_for('static', filename=f'images/{current_user.id}/{empty_space}')
    create_directory_if_not_exists(os.path.dirname(es_path))
    # Resize
    image = Image.open(es_path)
    target_size = (767, 498)  # Set your desired width and height
    resized_image = image.resize(target_size, Image.Resampling.LANCZOS)  # Use a resampling filter for better quality
    resized_image.save(es_path)

    run_preprocessor("seg_ofade20k", es_path)
    segmented_path = "disruptor" + url_for('static', filename=f'images/{current_user.id}/preprocessed/preprocessed.jpg')

    # Find the right empty space image
    from disruptor.green_screen.find_similar.ssim import compare
    import glob

    segmented_predefined = "disruptor/green_screen/find_similar/images/empty_space_segmented"
    segmented_paths = glob.glob(os.path.join(segmented_predefined, '*.png'))
    max_similarity = -1  # Initialize max_similarity to a value that's lower than any possible similarity score
    max_similar_es = ""
    for image_path in segmented_paths:
        similarity = compare(segmented_path, image_path)
        if similarity > max_similarity:
            max_similarity = similarity
            max_similar_es = image_path

    # Find corresponding staged image
    import re
    max_similar_stage = str(re.search(r'\d+', os.path.basename(max_similar_es)).group()) + "_staged.jpg"
    max_similar_stage_path = "disruptor/green_screen/find_similar/images/staged/" + max_similar_stage

    # Parse furniture from the selected staged image

    # Create masks
    run_preprocessor("seg_ofade20k", max_similar_stage_path)
    mask_dir = f"disruptor/static/images/{current_user.id}/parsed_furniture"
    create_directory_if_not_exists(mask_dir)
    # We update the directory, to get rid of the rubbish from the previous segmentations
    remove_files(mask_dir)
    parse_objects(f'disruptor/static/images/{current_user.id}/preprocessed/preprocessed.jpg', current_user.id)
    prepare_masks(current_user)

    # Create png foreground
    from disruptor.green_screen.preprocess.create_pngs import create_fg, overlay
    create_fg(mask_dir, max_similar_stage_path, current_user.id)
    fg_path = "disruptor" + url_for('static', filename=f"images/{current_user.id}/preprocessed/foreground.png")
    create_directory_if_not_exists(os.path.dirname(fg_path))
    overlay(es_path, fg_path, current_user.id)

    # Run SD to process it
    query = GreenScreenImageQuery(text)
    query.run()

    # style="current_image.jpg"
    # style_image_path = "disruptor" + url_for('static', filename=f'images/{style}')
    # run_preprocessor("seg_ofade20k", style_image_path)
    # parse_objects()
    #
    # furniture_dir = "disruptor/static/images/parsed_furniture"
    # groups_dir = "disruptor/static/images/parsed_furniture"
    # unite_groups(furniture_dir, groups_dir, [["bed", "blanket;cover", "cushion", "pillow"], ["table", "pot", "plant;flora;plant;life"]])
    #
    # prep_bg_image(empty_space)
    # prep_fg_image("bed_blanket;cover_cushion_pillow.jpg")
    # run_graconet()