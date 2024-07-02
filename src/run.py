import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sdquery import apply_style
from tools import create_directory_if_not_exists, save_encoded_image, get_encoded_image_from_path

app = Flask(__name__)
CORS(app)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", None)

IMAGE_DIRECTORY = 'images'
INPUT_IMAGE_NAME = 'user_image.png'
INPUT_IMAGE_PATH = os.path.join(IMAGE_DIRECTORY, INPUT_IMAGE_NAME)
OUTPUT_IMAGE_NAME ='applied.jpg'
OUTPUT_IMAGE_PATH = os.path.join(IMAGE_DIRECTORY, OUTPUT_IMAGE_NAME)


@app.route("/ai/get_insane_image_1337", methods=['POST'])
def get_insane_image_1337():
    data = request.get_json()
    room_choice = data.get('room_choice')
    style_budget_choice = data.get('style_budget_choice')
    input_image = data.get('input_image')

    create_directory_if_not_exists(IMAGE_DIRECTORY)
    save_encoded_image(input_image, INPUT_IMAGE_PATH)
    apply_style(INPUT_IMAGE_NAME, room_choice, style_budget_choice)

    output_image = get_encoded_image_from_path(OUTPUT_IMAGE_PATH)
    return jsonify({'output_image': output_image})


def is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


if __name__ == "__main__":
    if is_port_in_use(port=5000):
        app.run(host="0.0.0.0", port=5001, debug=False)
    else:
        app.run(host="0.0.0.0", debug=False)