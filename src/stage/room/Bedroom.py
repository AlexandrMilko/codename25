from stage.room.Room import Room
from tools import (calculate_angle_from_top_view, get_image_size,
                   convert_png_to_mask, overlay_masks, run_preprocessor, save_mask_of_size)
import numpy as np
import os
from math import radians
import time

from PIL import Image


class Bedroom(Room):

    def stage(self):
        camera_height, pitch_rad, roll_rad, height = self.prepare_empty_room_data()

        prerequisite_path = f'images/preprocessed/prerequisite.png'
        tmp_mask_path = f'images/preprocessed/furniture_piece_mask.png'
        segmented_es_path = f'images/preprocessed/seg_prerequisite.png'
        mask_path = f'images/preprocessed/furniture_mask.png'

        # Add curtains
        self.add_curtains(camera_height, (pitch_rad, roll_rad), mask_path, tmp_mask_path, prerequisite_path)

        # Add plant
        # TODO change algo for plant with new Kyrylo algorithm
        # self.add_plant((pitch_rad, roll_rad), mask_path, tmp_mask_path, prerequisite_path)

        # Add kitchen_table_with_chairs
        self.add_bed((pitch_rad, roll_rad), mask_path, tmp_mask_path, prerequisite_path)

        # Create windows mask for staged room
        run_preprocessor("seg_ofade20k", prerequisite_path, "seg_prerequisite.png", height)
        Room.save_windows_mask(segmented_es_path,
                               f'images/preprocessed/windows_mask_inpainting.png')

    def add_bed(self, camera_angles_rad: tuple, mask_path, tmp_mask_path, prerequisite_path):
        from stage.furniture.Bed import Bed
        from tools import convert_png_to_mask, image_overlay, overlay_masks
        pitch_rad, roll_rad = camera_angles_rad
        bed = Bed()
        wall = self.get_biggest_wall()
        render_directory = f'images/preprocessed'
        wall.save_mask(os.path.join(render_directory, 'wall_mask.png'))
        pixel_for_placing = bed.find_placement_pixel(os.path.join(render_directory, 'wall_mask.png'))
        print(f"BED placement pixel: {pixel_for_placing}")
        yaw_angle = wall.find_angle_from_3d(self, pitch_rad, roll_rad)
        render_parameters = (
            bed.calculate_rendering_parameters(self, pixel_for_placing, yaw_angle, (roll_rad, pitch_rad)))
        width, height = get_image_size(self.empty_room_image_path)
        render_parameters['resolution_x'] = width
        render_parameters['resolution_y'] = height
        bed_image = bed.request_blender_render(render_parameters)
        bed_image.save(tmp_mask_path)
        convert_png_to_mask(tmp_mask_path)
        overlay_masks(tmp_mask_path, mask_path, mask_path)
        background_image = Image.open(prerequisite_path)
        combined_image = image_overlay(bed_image, background_image)
        combined_image.save(prerequisite_path)
