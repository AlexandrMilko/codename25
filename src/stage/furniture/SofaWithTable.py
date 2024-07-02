from stage.furniture.Furniture import FloorFurniture
import cv2
import numpy as np

class SofaWithTable(FloorFurniture):
    # We use it to scale the model to metric units
    scale = 1, 1, 1
    # We use it to compensate the angle if the model is originally rotated in a wrong way
    default_angles = 0, 0, 0

    def __init__(self, model_path='3Ds/living_room/sofa_with_table.usdc'):
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