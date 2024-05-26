from disruptor.stage.Room import Room
from disruptor.stage.Floor import Floor
from disruptor.stage.FurniturePiece import FurniturePiece, Bed, Curtain, Plant
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
        print(roll, pitch, "ROLL and PITCH of the CAMERA")
        pitch_rad, roll_rad = radians(pitch), radians(roll)

        # Add time for Garbage Collector
        time.sleep(5)

        from disruptor.stage.DepthAnything.depth_estimation import image_pixels_to_depth
        image_pixels_to_depth(self.original_image_path)

        # Add time for Garbage Collector
        time.sleep(5)

        # from disruptor.stage.DepthAnything.depth_estimation import image_pixels_to_3d, rotate_3d_points
        # image_pixels_to_3d(self.original_image_path, "my_3d_space.txt")
        # rotate_3d_points("my_3d_space.txt", "my_3d_space_rotated.txt", -pitch_rad, -roll_rad)

        # Segment our empty space room. It is used in Room.save_windows_mask
        from disruptor.tools import get_image_size, run_preprocessor
        width, height = get_image_size(self.original_image_path)
        run_preprocessor("seg_ofade20k", self.original_image_path, current_user_id, "segmented_es.png", height)

        # Add curtains
        prerequisite_path = f'disruptor/static/images/{current_user_id}/preprocessed/prerequisite.png'
        curtain = Curtain()
        Room.save_windows_mask(f'disruptor/static/images/{current_user_id}/preprocessed/windows_mask.png',
                               current_user_id)
        pixels_for_placing = curtain.find_placement_pixel(
            f'disruptor/static/images/{current_user_id}/preprocessed/windows_mask.png')
        print(f"CURTAINS placement pixels: {pixels_for_placing}")
        Image.open(self.original_image_path).save(prerequisite_path)
        for window in pixels_for_placing:
            left_top_point, right_top_point = window
            yaw_angle = calculate_angle_from_top_view(*[self.infer_3d(pixel, pitch_rad, roll_rad) for
                                                        pixel in (left_top_point, right_top_point)])
            for pixel in (left_top_point, right_top_point):
                render_parameters = curtain.calculate_rendering_parameters(self, pixel, yaw_angle,
                                                                           (roll_rad, pitch_rad), current_user_id)
                width, height = get_image_size(self.original_image_path)
                render_parameters['resolution_x'] = width
                render_parameters['resolution_y'] = height
                curtain_image = curtain.request_blender_render(render_parameters)
                background_image = Image.open(prerequisite_path)
                combined_image = image_overlay(curtain_image, background_image)
                combined_image.save(prerequisite_path)

        # Add time for Garbage Collector
        time.sleep(5)

        # Add plant
        plant = Plant()
        seg_image_path = f'disruptor/static/images/{current_user_id}/preprocessed/segmented_es.png'
        save_path = 'disruptor/static/images/floor_mask.png'
        Floor.save_mask(seg_image_path, save_path)
        pixels_for_placing = plant.find_placement_pixel(save_path)
        print(f"PLANT placement pixels: {pixels_for_placing}")
        import random
        random_index = random.randint(0, len(pixels_for_placing) - 1)
        render_parameters = (
            plant.calculate_rendering_parameters(self, pixels_for_placing[random_index], (roll_rad, pitch_rad),
                                               current_user_id))
        width, height = get_image_size(self.original_image_path)
        render_parameters['resolution_x'] = width
        render_parameters['resolution_y'] = height
        plant_image = plant.request_blender_render(render_parameters)
        background_image = Image.open(prerequisite_path)
        combined_image = image_overlay(plant_image, background_image)
        combined_image.save(prerequisite_path)

        # Add time for Garbage Collector
        time.sleep(5)

        # Add Bed
        bed = Bed()
        wall = self.get_biggest_wall(current_user_id)
        render_directory = f'disruptor/static/images/{current_user_id}/preprocessed/furniture_render'
        wall.save_mask(os.path.join(render_directory, 'wall_mask.png'))
        pixel_for_placing = bed.find_placement_pixel(os.path.join(render_directory, 'wall_mask.png'))
        print(f"BED placement pixel: {pixel_for_placing}")
        yaw_angle = wall.find_angle_from_3d(self, pitch_rad, roll_rad)
        render_parameters = (bed.calculate_rendering_parameters(self, pixel_for_placing, yaw_angle, (roll_rad, pitch_rad), current_user_id))
        width, height = get_image_size(self.original_image_path)
        render_parameters['resolution_x'] = width
        render_parameters['resolution_y'] = height
        bed_image = bed.request_blender_render(render_parameters)
        background_image = Image.open(prerequisite_path)
        combined_image = image_overlay(bed_image, background_image)
        combined_image.save(prerequisite_path)