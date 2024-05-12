from disruptor.stage.Room import Room
from disruptor.stage.FurniturePiece import FurniturePiece, Bed, Curtain
from disruptor.tools import calculate_angle_from_top_view
import numpy as np
import os
from math import radians


class Bedroom(Room):

    def stage(self, current_user_id):
        roll, pitch = np.negative(np.degrees(self.find_roll_pitch(current_user_id)))
        walls = self.get_walls(current_user_id)
        compensate_pitch = -radians(pitch)
        compensate_roll = -radians(roll)

        # Add Bed
        bed = Bed()
        wall = walls[0]
        render_directory = f'disruptor/static/images/{current_user_id}/preprocessed/furniture_render'
        wall.save_mask(os.path.join(render_directory, 'wall_mask.png'))
        pixels_for_placing = bed.find_placement_pixel(os.path.join(render_directory, 'wall_mask.png'))
        yaw_angle = wall.find_angle_from_3d(self, compensate_pitch, compensate_roll)
        for pixel in pixels_for_placing:
            bed.calculate_rendering_parameters(self, pixel, yaw_angle, (roll, pitch), current_user_id)

        # Add curtains
        curtain = Curtain()
        Room.save_windows_mask(f'disruptor/static/images/{current_user_id}/preprocessed/windows_mask.png',
                               current_user_id)
        pixels_for_placing = curtain.find_placement_pixel(
            f'disruptor/static/images/{current_user_id}/preprocessed/windows_mask.png')
        yaw_angle = calculate_angle_from_top_view(*[self.infer3d(pixel, compensate_pitch, compensate_roll) for pixel in pixels_for_placing])
        for pixel in pixels_for_placing:
            curtain.calculate_rendering_parameters(self, pixel, yaw_angle, (roll, pitch), current_user_id)