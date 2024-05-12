import os.path
import cv2
import numpy as np

from disruptor.stage import Room
from disruptor.tools import get_filename_without_extension, create_directory_if_not_exists
# from vedo import *


class FurniturePiece:
    scale = 1, 1, 1
    default_angles = 0, 0, 0

    def __init__(self, model_path, wall_projection_model_path, floor_projection_model_path):
        (
            self.model_path,
            self.wall_projection_model_path,
            self.floor_projection_model_path,
        ) = (
            model_path,
            wall_projection_model_path,
            floor_projection_model_path,
        )

    def get_scale(self):
        return self.scale

    def get_default_angles(self):
        return self.default_angles


class Bed(FurniturePiece):
    # We use it to scale the model to metric units
    scale = 0.01, 0.01, 0.01
    # We use it to compensate the angle if the model is originally rotated in a wrong way
    default_angles = 0, 0, 90

    def __init__(self, model_path='disruptor/stage/3Ds/bedroom/bed/bed.obj',
                 wall_projection_model_path='disruptor/stage/3Ds/bedroom/bed/bed_back.obj',
                 floor_projection_model_path='disruptor/stage/3Ds/bedroom/bed/bed_bottom.obj',
                 ):
        super().__init__(model_path, wall_projection_model_path, floor_projection_model_path)

    @staticmethod
    def find_placement_pixel(wall_mask_path):
        wall_mask = cv2.imread(wall_mask_path, cv2.IMREAD_GRAYSCALE)
        wall_contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate offset from bed centroid to wall centroid
        wall_centroid = np.mean(wall_contours[0], axis=0)[0]
        pixel_x = wall_centroid[0]
        pixel_y = wall_centroid[1]

        return int(pixel_x), int(pixel_y)

class Curtain(FurniturePiece):
    pass