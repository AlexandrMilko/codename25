from PIL import Image
import os
from disruptor.tools import move_file, run_preprocessor, copy_file, convert_to_mask, find_bed_placement_coordinates, overlay_images, create_furniture_mask
from disruptor.stage.FurniturePiece import FurniturePiece, Bed
from disruptor.stage.Wall import Wall
import numpy as np

class Room:
    # BGR, used in segmented images
    window_color = (230, 230, 230)
    door_color = (51, 255, 8)
    floor_color = (50, 50, 80)
    def __init__(self, original_image_path): # Original image path is an empty space image
        self.original_image_path = original_image_path

    def find_roll_pitch(self, current_user_id) -> tuple[float, float]:
        es_img = Image.open(self.original_image_path)
        width, height = es_img.size
        es_img.close()
        run_preprocessor("normal_bae", self.original_image_path, current_user_id, "users.png", height)
        copy_file(self.original_image_path, "disruptor/UprightNet/imgs/rgb/users.png") # We copy it because we will use it later in get_wall method and we want to have access to the image
        move_file(f"disruptor/static/images/{current_user_id}/preprocessed/users.png",
                  "disruptor/UprightNet/imgs/normal_pair/users.png")
        from disruptor.UprightNet.infer import get_roll_pitch
        return get_roll_pitch()

    def get_walls(self, current_user_id):
        es_img = Image.open(self.original_image_path)
        width, height = es_img.size
        es_img.close()
        run_preprocessor("seg_ofade20k", self.original_image_path, current_user_id, "segmented_es.png", height)
        from disruptor.stage.Wall import Wall
        return Wall.find_walls(f'disruptor/static/images/{current_user_id}/preprocessed/segmented_es.png')

    def stage(self, text_parameters, current_user_id):
        room_type = text_parameters.split(", ")[1].lower()
        roll, pitch = np.negative(np.degrees(self.find_roll_pitch(current_user_id)))
        walls = self.get_walls(current_user_id)
        if room_type == "bedroom":
            furniture_pieces = [Bed()]
        elif room_type == "living room":
            pass
        else:
            furniture_pieces = []

        for i in range(len(walls)):
            try:
                self.add_furniture(furniture_pieces[i], walls[i], (roll, pitch), current_user_id)
            except IndexError:
                break


    def add_furniture(self, furniture: FurniturePiece, wall: Wall, camera_angles: tuple[float, float], current_user_id): # Saves corresponding mask and render
        roll, pitch = camera_angles
        yaw = wall.find_angle()

        furniture.render_model(f'disruptor/static/images/{current_user_id}/preprocessed/furniture_render', (roll, yaw, pitch))
        render_directory = f'disruptor/static/images/{current_user_id}/preprocessed/furniture_render'
        wall.save_mask(os.path.join(render_directory, 'wall_mask.png'))
        for filename in os.listdir(render_directory):
            if 'back' in filename or 'bottom' in filename:
                convert_to_mask(os.path.join(render_directory, filename))
        coords_for_placing = find_bed_placement_coordinates(os.path.join(render_directory, 'bed_back.png'), os.path.join(render_directory, 'wall_mask.png'), render_directory)
        create_furniture_mask(self.original_image_path, [os.path.join(render_directory, 'bed.png')], [coords_for_placing], f'disruptor/static/images/{current_user_id}/preprocessed/inpainting_mask.png')
        overlay_images(os.path.join(render_directory, 'bed_back.png'), os.path.join(render_directory, 'wall_mask.png'), f'disruptor/static/images/{current_user_id}/preprocessed/test.png', coords_for_placing)
        overlay_images(os.path.join(render_directory, 'bed.png'), self.original_image_path, f'disruptor/static/images/{current_user_id}/preprocessed/prerequisite.jpg', coords_for_placing)
        # convert to masks
        # scale the renders to 0.4 height of wall height(write get_height in Wall)
        # write a function that adds an image to a specific location in another image (use create_pngs.py and preprocess_for_empty_space.py)
        # add render onto your prerequisite which is original_image_copy in the beginning
        # write a function that positions it properly