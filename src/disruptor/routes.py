from flask import url_for, request, jsonify
from disruptor.sdquery import TextQuery, ImageQuery, ControlNetImageQuery, apply_style
from disruptor import app
@app.route("/ai/get_insane_image_1337", methods=['POST'])
def get_insane_image_1337():
    data = request.get_json()
    room_choice = data.get('room_choice')
    style_budget_choice = data.get('style_budget_choice')
    input_image = data.get('input_image')

    filename = 'user_image.png'
    directory = f"disruptor/images/"
    from disruptor.tools import create_directory_if_not_exists, save_encoded_image, get_encoded_image
    create_directory_if_not_exists(directory)
    file_path = directory + filename
    save_encoded_image(input_image, file_path)

    apply_style(filename, room_choice, style_budget_choice)

    return jsonify({'output_image': get_encoded_image('disruptor' + url_for('images', filename=f'applied.jpg'))})