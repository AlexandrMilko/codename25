from tools import create_directory_if_not_exists, save_encoded_image, get_encoded_image_from_path
from flask import Flask, request, jsonify
from flask_cors import CORS
from sdquery import apply_style
from constants import Path
import os


app = Flask(__name__)
CORS(app)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", None)


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


def is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


if __name__ == "__main__":
    if is_port_in_use(port=5000):
        app.run(host="0.0.0.0", port=5001, debug=False)
    else:
        app.run(host="0.0.0.0", debug=False)