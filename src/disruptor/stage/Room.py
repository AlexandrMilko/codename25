import cv2
from PIL import Image
import os
from disruptor.tools import move_file, run_preprocessor, copy_file, convert_to_mask, overlay_images, create_furniture_mask
from disruptor.stage.FurniturePiece import FurniturePiece
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

    # def stage(self, text_parameters, current_user_id):
    #     room_type = text_parameters.split(", ")[1].lower()
    #     roll, pitch = np.negative(np.degrees(self.find_roll_pitch(current_user_id)))
    #     walls = self.get_walls(current_user_id)
    #     if room_type == "bedroom":
    #         furniture_pieces = [Bed()]
    #     elif room_type == "living room":
    #         pass
    #     else:
    #         furniture_pieces = []
    #
    #     for i in range(len(walls)):
    #         try:
    #             self.add_furniture(furniture_pieces[i], walls[i], (roll, pitch), current_user_id)
    #         except IndexError:
    #             break


    def add_furniture(self, furniture: FurniturePiece, placement_pixel: tuple[int, int], yaw_angle: float, camera_angles: tuple[float, float], current_user_id): # Saves corresponding mask and render
        from math import radians
        roll, pitch = camera_angles
        compensate_pitch = -radians(pitch)
        compensate_roll = -radians(roll)
        default_angles = furniture.get_default_angles()

        obj_offsets = self.infer_3d(placement_pixel, compensate_pitch, compensate_roll) # We set negative rotation to compensate
        obj_angles = radians(default_angles[0]), radians(default_angles[1]), radians(default_angles[2] + yaw_angle)  # In blender, yaw angle is around z axis. z axis is to the top
        obj_scale = furniture.get_scale()
        # We set opposite
        camera_angles = radians(90) + compensate_pitch, -compensate_roll, 0 # We add 90 to the pitch, because originally camera is rotated pointing downwards in Blender
        camera_height = self.estimate_camera_height((compensate_pitch, compensate_roll), current_user_id)
        camera_location = 0, 0, camera_height
        obj_offsets_floor = obj_offsets.copy()
        obj_offsets_floor[2] = 0

        print("Furniture coords")
        print(obj_offsets, "obj_offsets")
        print(obj_offsets_floor, "obj_offsets for blender with floor z axis")
        print(obj_angles, "obj_angles")
        print(yaw_angle, "yaw_angle")
        print(obj_scale, "obj_scale")
        print(camera_angles, "camera_angles")
        print(camera_location, "camera_location")
        # furniture.render_model(f'disruptor/static/images/{current_user_id}/preprocessed/furniture_render', (roll, yaw, pitch))
        # for filename in os.listdir(render_directory):
        #     if 'back' in filename or 'bottom' in filename:
        #         convert_to_mask(os.path.join(render_directory, filename))
        # # coords_for_placing = find_bed_placement_coordinates(os.path.join(render_directory, 'bed_back.png'), os.path.join(render_directory, 'wall_mask.png'), render_directory)
        # create_furniture_mask(self.original_image_path, [os.path.join(render_directory, 'bed.png')], [coords_for_placing], f'disruptor/static/images/{current_user_id}/preprocessed/inpainting_mask.png')
        # overlay_images(os.path.join(render_directory, 'bed_back.png'), os.path.join(render_directory, 'wall_mask.png'), f'disruptor/static/images/{current_user_id}/preprocessed/test.png', coords_for_placing)
        # overlay_images(os.path.join(render_directory, 'bed.png'), self.original_image_path, f'disruptor/static/images/{current_user_id}/preprocessed/prerequisite.jpg', coords_for_placing)
        # # convert to masks
        # # scale the renders to 0.4 height of wall height(write get_height in Wall)
        # # write a function that adds an image to a specific location in another image (use create_pngs.py and preprocess_for_empty_space.py)
        # # add render onto your prerequisite which is original_image_copy in the beginning
        # # write a function that positions it properly

    def infer_3d(self, pixel: tuple[int, int], compensate_pitch_rad: float, compensate_roll_rad: float):
        from disruptor.stage.depth_estimation import image_pixel_to_3d, rotate_3d_point
        target_point = image_pixel_to_3d(self.original_image_path, pixel)
        # We rotate it back to compensate our camera rotation
        offset_relative_to_camera = rotate_3d_point(target_point, compensate_pitch_rad, compensate_roll_rad)
        return offset_relative_to_camera

    def estimate_camera_height(self, compensate_angles: tuple[float, float], current_user_id):
        compensate_pitch, compensate_roll = compensate_angles
        points_filepath = f'disruptor/static/images/{current_user_id}/preprocessed/3d_coords.txt'
        rotated_points_filepath = f'disruptor/static/images/{current_user_id}/preprocessed/3d_coords_rotated.txt'
        from disruptor.stage.depth_estimation import rotate_3d_points, image_pixels_to_3d
        from disruptor.tools import find_abs_min_z
        image_pixels_to_3d(self.original_image_path,
                           f'disruptor/static/images/{current_user_id}/preprocessed/3d_coords.txt')
        rotated_points = rotate_3d_points(points_filepath, rotated_points_filepath, compensate_pitch, compensate_roll)
        camera_height = find_abs_min_z(rotated_points)
        return camera_height

    @staticmethod
    def find_number_of_windows(windows_mask_path: str) -> int:
        # Загрузка изображения
        img = cv2.imread(windows_mask_path, cv2.IMREAD_GRAYSCALE)  # Укажите правильный путь к файлу
        # Адаптивная пороговая обработка
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        # Морфологические операции для разделения окон
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)  # Увеличьте iterations, если нужно
        # Поиск контуров
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Визуализация результатов
        img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Переводим в цвет для визуализации контуров
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)
        # Подсчет количества окон
        number_of_windows = len(contours)

        return number_of_windows