from disruptor.stage.Room import Room
from disruptor.stage.FurniturePiece import FurniturePiece, Bed
import numpy as np
import os
from math import radians


class Bedroom(Room):
    furniture_pieces = [Bed()]

    def stage(self, current_user_id):
        roll, pitch = np.negative(np.degrees(self.find_roll_pitch(current_user_id)))
        walls = self.get_walls(current_user_id)
        compensate_pitch = -radians(pitch)
        compensate_roll = -radians(roll)

        for i in range(len(walls)):
            try:
                furniture_piece = self.furniture_pieces[i]
                wall = walls[i]
                render_directory = f'disruptor/static/images/{current_user_id}/preprocessed/furniture_render'
                wall.save_mask(os.path.join(render_directory, 'wall_mask.png'))
                pixels_for_placing = furniture_piece.find_placement_pixel(os.path.join(render_directory, 'wall_mask.png'))
                yaw_angle = wall.find_angle_from_3d(self, compensate_pitch, compensate_roll)
                for pixel in pixels_for_placing:
                    self.add_furniture(furniture_piece, pixel, yaw_angle, (roll, pitch), current_user_id)
            except IndexError:
                break
