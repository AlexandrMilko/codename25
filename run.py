import os

from flask import Flask, request, jsonify
from flask_cors import CORS

from constants import Config, Path
from tools import create_visuals_dir, save_encoded_image, get_encoded_image_from_path

app = Flask(__name__)
CORS(app)


@app.route("/ai/get_insane_image_1337", methods=['POST'])
def get_insane_image_1337():
    data = request.get_json()
    room_choice = data.get('room_choice')
    style_budget_choice = data.get('style_budget_choice')
    input_image = data.get('input_image')

    create_visuals_dir()
    save_encoded_image(input_image, Path.INPUT_IMAGE.value)
    output_image_paths = apply_style(Path.INPUT_IMAGE.value, room_choice, style_budget_choice)

    if Config.DO_POSTPROCESSING.value:
        # WARNING! WE REMOVED WEBUI, so NOW THE POSTPROCESSING IS DONE THROUGH COMFYUI AND IT DOES NOT MAKE IMAGE BETTER
        # UNTIL we decide what we do with postprocessing - set DO_POSTPROCESSING to FALSE.
        # Plus, at this point, postprocessing is run only for one image
        output_image = get_encoded_image_from_path(Path.OUTPUT_IMAGE.value)
        return jsonify({'output_image': output_image})
    else:
        encoded_images_dict = dict()
        for i in range(len(output_image_paths)):
            image_path = output_image_paths[i]
            encoded_images_dict[f"output_image_{i}"] = get_encoded_image_from_path(image_path)
        return jsonify(encoded_images_dict)


def apply_style(es_path, room_choice, style_budget_choice):
    import stage

    if room_choice.lower() == "bedroom":
        room = stage.Bedroom(es_path)
        output_image_paths = room.stage()
    elif room_choice.lower() == "kitchen":
        room = stage.Kitchen(es_path)
        output_image_paths = room.stage()
    elif room_choice.lower() == "living room":
        room = stage.LivingRoom(es_path)
        output_image_paths = room.stage()
    else:
        raise Exception(f"Wrong Room Type was specified: {room_choice.lower()}")
    return output_image_paths

def is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


if __name__ == "__main__":
    available_port = 5001 if is_port_in_use(port=5000) else 5000
    app.run(host="0.0.0.0", port=available_port, debug=False)
