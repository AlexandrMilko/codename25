from stage.Room import Room
from stage.Floor import Floor
from stage.Furniture import Furniture, Bed, Curtain, Plant
from tools import calculate_angle_from_top_view, get_image_size, create_mask_of_size, convert_png_to_mask, overlay_masks
import numpy as np
import os
from math import radians
import time

from PIL import Image

class Bedroom(Room):

    def stage(self):
        roll, pitch = np.negative(np.degrees(self.find_roll_pitch()))
        print(roll, pitch, "ROLL and PITCH of the CAMERA")
        pitch_rad, roll_rad = radians(pitch), radians(roll)

        # Add time for Garbage Collector
        time.sleep(5)

        from stage.DepthAnything.depth_estimation import image_pixels_to_point_cloud, depth_ply_path, depth_npy_path
        image_pixels_to_point_cloud(self.empty_room_image_path)
        floor_layout_path = 'images/preprocessed/floor_layout.png'
        self.save_floor_layout_image(depth_ply_path, depth_npy_path, floor_layout_path)

        # Add time for Garbage Collector
        time.sleep(5)

        # from stage.DepthAnything.depth_estimation import image_pixels_to_3d, rotate_3d_points
        # image_pixels_to_3d(self.empty_room_image_path, "my_3d_space.txt")
        # rotate_3d_points("my_3d_space.txt", "my_3d_space_rotated.txt", -pitch_rad, -roll_rad)

        # Segment our empty space room. It is used in Room.save_windows_mask
        from tools import get_image_size, run_preprocessor
        width, height = get_image_size(self.empty_room_image_path)
        run_preprocessor("seg_ofade20k", self.empty_room_image_path, "segmented_es.png", height)

        camera_height = self.estimate_camera_height([pitch_rad, roll_rad])

        # Create an empty mask of same size as image
        mask_path = f'images/preprocessed/furniture_mask.png'
        tmp_mask_path = f'images/preprocessed/furniture_piece_mask.png'
        width, height = get_image_size(self.empty_room_image_path)
        empty_mask = create_mask_of_size(width, height)
        print("Saving empty mask to:", mask_path)
        empty_mask.save(mask_path)
        print("Empty mask saved successfully!")

        prerequisite_path = f'images/preprocessed/prerequisite.png'

        # Add curtains
        self.add_curtains(camera_height, (pitch_rad, roll_rad), mask_path, tmp_mask_path, prerequisite_path)

        # Add time for Garbage Collector
        time.sleep(5)

        # Add plant
        self.add_plant((pitch_rad, roll_rad), mask_path, tmp_mask_path, prerequisite_path)

        # Add time for Garbage Collector
        time.sleep(5)

        # Add Bed
        self.add_bed(camera_height, (pitch_rad, roll_rad), mask_path, tmp_mask_path, prerequisite_path)

        # Create windows mask for staged room
        run_preprocessor("seg_ofade20k", prerequisite_path, "seg_prerequisite.png", height)
        segmented_es_path = f'images/preprocessed/seg_prerequisite.png'
        Room.save_windows_mask(segmented_es_path,
                               f'images/preprocessed/windows_mask_inpainting.png')