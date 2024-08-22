from .Furniture import FloorFurniture
from constants import Path
import numpy as np
import cv2


class Bed(FloorFurniture):
    default_angles = 0, 0, 90
    def __init__(self, model_path=Path.BED_MODEL.value):
        super().__init__(model_path)

    @staticmethod
    def find_placement_pixel(wall_mask_path) -> tuple[int, int]:
        wall_mask = cv2.imread(wall_mask_path, cv2.IMREAD_GRAYSCALE)
        wall_contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate offset from bed centroid to wall centroid
        wall_centroid = np.mean(wall_contours[0], axis=0)[0]
        pixel_x = wall_centroid[0]
        pixel_y = wall_centroid[1]

        return int(pixel_x), int(pixel_y)
