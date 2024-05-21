from disruptor.stage.Room import Room
from disruptor.stage.FurniturePiece import FurniturePiece, Bed, Curtain
from disruptor.tools import calculate_angle_from_top_view, get_image_size
import numpy as np
import os
from math import radians
import time

from PIL import Image
from disruptor.tools import image_overlay


class Bedroom(Room):

    def stage(self, current_user_id):
        roll, pitch = np.negative(np.degrees(self.find_roll_pitch(current_user_id)))
        walls = self.get_walls(current_user_id)
        roll, pitch = 0, 0 # WARNING, remove it to use camera angle estimation
        compensate_pitch = -radians(pitch)
        compensate_roll = -radians(roll)

        # Add time for Garbage Collector
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)

        from disruptor.stage.DepthAnything.depth_estimation import image_pixels_to_depth
        image_pixels_to_depth(self.original_image_path)

        # Add time for Garbage Collector
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)

        # from disruptor.stage.DepthAnything.depth_estimation import image_pixels_to_3d, rotate_3d_points
        # image_pixels_to_3d(self.original_image_path, "my_3d_space.txt")
        # rotate_3d_points("my_3d_space.txt", "my_3d_space_rotated.txt", compensate_pitch, compensate_roll)

        # Add Bed
        bed = Bed()
        wall = walls[0]
        render_directory = f'disruptor/static/images/{current_user_id}/preprocessed/furniture_render'
        wall.save_mask(os.path.join(render_directory, 'wall_mask.png'))
        pixels_for_placing = bed.find_placement_pixel(os.path.join(render_directory, 'wall_mask.png'))
        print(f"BED placement pixel: {pixels_for_placing}")
        yaw_angle = wall.find_angle_from_3d(self, compensate_pitch, compensate_roll)
        for pixel in pixels_for_placing:
            render_parameters = (bed.calculate_rendering_parameters(self, pixel, yaw_angle, (roll, pitch), current_user_id))
            width, height = get_image_size(self.original_image_path)
            render_parameters['resolution_x'] = width
            render_parameters['resolution_y'] = height
            furniture_image = bed.request_blender_render(render_parameters)
            background_image = Image.open(self.original_image_path)
            combined_image = image_overlay(furniture_image, background_image)
            combined_image.save(f'disruptor/static/images/{current_user_id}/preprocessed/prerequisite.png')

        # Add time for Garbage Collector
        time.sleep(5)

        # Add curtains
        curtain = Curtain()
        Room.save_windows_mask(f'disruptor/static/images/{current_user_id}/preprocessed/windows_mask.png',
                               current_user_id)
        pixels_for_placing = curtain.find_placement_pixel(
            f'disruptor/static/images/{current_user_id}/preprocessed/windows_mask.png')
        print(f"CURTAINS placement pixels: {pixels_for_placing}")
        for window in pixels_for_placing:
            left_top_point, right_top_point = window
            yaw_angle = calculate_angle_from_top_view(*[self.infer_3d(pixel, compensate_pitch, compensate_roll) for
                                                        pixel in (left_top_point, right_top_point)])
            for pixel in (left_top_point, right_top_point):
                curtain.calculate_rendering_parameters(self, pixel, yaw_angle, (roll, pitch), current_user_id)
