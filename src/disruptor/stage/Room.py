from PIL import Image
import os
from disruptor.tools import move_file, run_preprocessor, copy_file
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