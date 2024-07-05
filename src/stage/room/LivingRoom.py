from stage.room.Room import Room
from tools import calculate_angle_from_top_view, get_image_size, create_mask_of_size, convert_png_to_mask, overlay_masks
import numpy as np
import os
from math import radians
import time

from PIL import Image
from tools import image_overlay


class LivingRoom(Room):

    def stage(self):
        roll, pitch = np.negative(np.degrees(self.find_roll_pitch()))
        print(roll, pitch, "ROLL and PITCH of the CAMERA")
        pitch_rad, roll_rad = radians(pitch), radians(roll)

        # Add time for Garbage Collector
        time.sleep(1)

        from DepthAnything.depth_estimation import image_pixels_to_point_cloud, depth_ply_path, depth_npy_path
        image_pixels_to_point_cloud(self.empty_room_image_path)
        floor_layout_path = 'images/preprocessed/floor_layout.png'
        self.save_floor_layout_image(depth_ply_path, depth_npy_path, floor_layout_path)

        # Add time for Garbage Collector
        time.sleep(1)

        # from DepthAnything.depth_estimation import image_pixels_to_3d, rotate_3d_points
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
        time.sleep(1)

        # Add plant
        self.add_plant((pitch_rad, roll_rad), mask_path, tmp_mask_path, prerequisite_path)

        # Add time for Garbage Collector
        time.sleep(1)

        # Add SofaWithTable
        self.add_sofa_with_table((pitch_rad, roll_rad), mask_path, tmp_mask_path, prerequisite_path)

        # Create windows mask for staged room
        run_preprocessor("seg_ofade20k", prerequisite_path, "seg_prerequisite.png", height)
        segmented_es_path = f'images/preprocessed/seg_prerequisite.png'
        Room.save_windows_mask(segmented_es_path,
                               f'images/preprocessed/windows_mask_inpainting.png')

    # def add_sofa_with_table(self, camera_angles_rad: tuple, mask_path, tmp_mask_path, prerequisite_path):
    #     from stage.furniture.SofaWithTable import SofaWithTable
    #     from tools import convert_png_to_mask, image_overlay, overlay_masks
    #     pitch_rad, roll_rad = camera_angles_rad
    #     sofa_with_table = SofaWithTable()
    #     wall = self.get_biggest_wall()
    #     render_directory = f'images/preprocessed/'
    #     wall.save_mask(os.path.join(render_directory, 'wall_mask.png'))
    #     pixel_for_placing = sofa_with_table.find_placement_pixel(os.path.join(render_directory, 'wall_mask.png'))
    #     print(f"SofaWithTable placement pixel: {pixel_for_placing}")
    #     yaw_angle = wall.find_angle_from_3d(self, pitch_rad, roll_rad)
    #     render_parameters = (
    #         sofa_with_table.calculate_rendering_parameters(self, pixel_for_placing, yaw_angle, (roll_rad, pitch_rad)))
    #     width, height = get_image_size(self.empty_room_image_path)
    #     render_parameters['resolution_x'] = width
    #     render_parameters['resolution_y'] = height
    #     sofa_image = sofa_with_table.request_blender_render(render_parameters)
    #     sofa_image.save(tmp_mask_path)
    #     convert_png_to_mask(tmp_mask_path)
    #     overlay_masks(tmp_mask_path, mask_path, mask_path)
    #     background_image = Image.open(prerequisite_path)
    #     combined_image = image_overlay(sofa_image, background_image)
    #     combined_image.save(prerequisite_path)
