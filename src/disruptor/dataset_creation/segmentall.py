import os
import base64
import json
import requests
import cv2

input_dir = "prodDataset/awesome/office/original"
output_dir = "prodDataset/awesome/office/es_segmented"
def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
def save_encoded_image(b64_image: str, output_path: str):
    """
    Save the given image to the given output path.
    """
    with open(output_path, "wb") as image_file:
        image_file.write(base64.b64decode(b64_image))
def submit_post(url: str, data: dict):
    """
    Submit a POST request to the given URL with the given data.
    """
    return requests.post(url, data=json.dumps(data))
def get_encoded_image(image_path):
    img = cv2.imread(image_path)
    # Encode into PNG and send to ControlNet
    retval, bytes = cv2.imencode('.png', img)
    return base64.b64encode(bytes).decode('utf-8')

def run_preprocessor(preprocessor_name, image_path): # TODO rename, i am not using it outside dataset creation. But there is another one with such name
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
    filename = os.path.basename(image_path)
    output_filepath = os.path.join(output_dir, filename)

    # If there was no such dir, we create it and try again
    try:
        save_encoded_image(response.json()['images'][0], output_filepath)
    except FileNotFoundError as e:
        create_directory_if_not_exists(output_dir)
        save_encoded_image(response.json()['images'][0], output_filepath)

if __name__ == "__main__":
    # Use os.listdir to get a list of filenames in the directory
    file_names = os.listdir(input_dir)

    # Convert the filenames to absolute paths
    file_paths = [os.path.join(input_dir, file) for file in file_names]

    # Now, file_paths contains the absolute paths to all files in the directory
    for file_path in file_paths:
        if "Before" in os.path.basename(file_path):
            run_preprocessor("seg_ofade20k", file_path)