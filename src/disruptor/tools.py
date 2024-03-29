import os
import cv2
import base64
import requests
import json
import shutil

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def min_max_scale(value, min_val, max_val):
    if max_val == min_val:
        return 0.5  # Handle division by zero (if min_val == max_val)
    return (value - min_val) / (max_val - min_val)

def move_file(src_path, dest_path):
    try:
        os.rename(src_path, dest_path)
        print(f"File moved successfully from '{src_path}' to '{dest_path}'.")
    except Exception as e:
        print(os.getcwd())
        print(f"Error: {e}")

def copy_file(src_path, dest_path):
    try:
        shutil.copy2(src_path, dest_path)
        print(f"File copied successfully from '{src_path}' to '{dest_path}'.")
    except Exception as e:
        print(f"Error: {e}")

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

def get_encoded_image(image_path):
    img = cv2.imread(image_path)
    # Encode into PNG and send to ControlNet
    retval, bytes = cv2.imencode('.png', img)
    return base64.b64encode(bytes).decode('utf-8')
def run_preprocessor(preprocessor_name, image_path, current_user_id, filename="preprocessed.jpg", res=512):
    input_image = get_encoded_image(image_path)
    data = {
        "controlnet_module": preprocessor_name,
        "controlnet_input_images": [input_image],
        "controlnet_processor_res": res,
        "controlnet_threshold_a": 64,
        "controlnet_threshold_b": 64
    }
    preprocessor_url = 'http://127.0.0.1:7861/controlnet/detect'
    response = submit_post(preprocessor_url, data)
    output_dir = f"disruptor/static/images/{current_user_id}/preprocessed"
    output_filepath = os.path.join(output_dir, filename)

    # If there was no such dir, we create it and try again
    try:
        save_encoded_image(response.json()['images'][0], output_filepath)
    except FileNotFoundError as e:
        create_directory_if_not_exists(output_dir)
        save_encoded_image(response.json()['images'][0], output_filepath)