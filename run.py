import os
import time

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

from constants import Config, Path
from tools import create_directory_if_not_exists, save_encoded_image, get_encoded_image_from_path, submit_post

app = Flask(__name__)
CORS(app)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", None)


def get_sd_domain():  # We use this function to check if Stable Diffusion is running in docker or on host system
    try:
        data = {"sd_model_checkpoint": "realisticVisionV60B1_v51HyperVAE.safetensors"}
        options_url = 'http://127.0.0.1:7861/sdapi/v1/options'
        response = submit_post(options_url, data)
        return "127.0.0.1"
    except requests.exceptions.ConnectionError:
        print("INFO: Using host.docker.internal for SD")
        return "host.docker.internal"


SD_DOMAIN = get_sd_domain()


@app.route("/ai/get_insane_image_1337", methods=['POST'])
def get_insane_image_1337():
    data = request.get_json()
    room_choice = data.get('room_choice')
    style_budget_choice = data.get('style_budget_choice')
    input_image = data.get('input_image')

    create_directory_if_not_exists(Path.IMAGES_DIR.value)
    save_encoded_image(input_image, Path.INPUT_IMAGE.value)
    apply_style(Path.INPUT_IMAGE.value, room_choice, style_budget_choice)

    output_image = get_encoded_image_from_path(Path.OUTPUT_IMAGE.value)
    return jsonify({'output_image': output_image})


def apply_style(es_path, room_choice, style_budget_choice):
    import stage

    if room_choice.lower() == "bedroom":
        room = stage.Bedroom(es_path)
        room.stage()
    elif room_choice.lower() == "kitchen":
        room = stage.Kitchen(es_path)
        room.stage()
    elif room_choice.lower() == "living room":
        room = stage.LivingRoom(es_path)
        room.stage()
    else:
        raise Exception(f"Wrong Room Type was specified: {room_choice.lower()}")

    if Config.DO_POSTPROCESSING.value and Config.UI.value == "webui":
        from postprocessing.postProcessingWebui import GreenScreenImageQuery
        from tools import restart_stable_diffusion
        import requests
        style, budget = style_budget_choice.split(", ")
        text = f"Residential, {room_choice}, {budget}, {style}"
        query = GreenScreenImageQuery(text)

        import torch
        try:
            query.run()
        except (torch.cuda.OutOfMemoryError, KeyError) as e:
            print(e, "RESTARTING THE STABLE DIFFUSION AND TRYING AGAIN!")
            restart_stable_diffusion(f'http://{SD_DOMAIN}:7861')
            query.run()

        # We restart it to deallocate memory. TODO fix it.
        try:
            time.sleep(3)
            restart_stable_diffusion(f'http://{SD_DOMAIN}:7861')
        except requests.exceptions.ConnectionError:
            print("Stable Diffusion restarting")


def is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


if __name__ == "__main__":
    available_port = 5001 if is_port_in_use(port=5000) else 5000
    app.run(host="0.0.0.0", port=available_port, debug=False)
