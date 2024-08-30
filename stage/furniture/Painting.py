import cv2
import numpy as np

from .Furniture import HangingFurniture
from constants import Path

class Painting(HangingFurniture):
    def __init__(self, model_path=Path.PAINTING_MODEL.value):
        super().__init__(model_path)

    @staticmethod
    def find_placement_pixel(image_path):
        mask, image = preprocess_image(image_path)
        black_white_image = create_black_white_image(mask)
        optimal_point = find_optimal_point(black_white_image)

        if not optimal_point:
            return None

        left_boundary, right_boundary = find_boundaries(black_white_image, optimal_point)

        if left_boundary and right_boundary:
            center_point = (int((left_boundary + right_boundary) // 2), int(optimal_point[1]))
            return left_boundary, center_point, right_boundary

        return None

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    target_color_hsv = np.array([0, 0, 120])
    lower_bound = target_color_hsv - np.array([10, 10, 10])
    upper_bound = target_color_hsv + np.array([10, 10, 10])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    return mask, image

def create_black_white_image(mask):
    black_white_image = np.zeros(mask.shape, dtype=np.uint8)
    black_white_image[mask > 0] = 255
    return black_white_image

def find_optimal_point(black_white_image):
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(black_white_image, connectivity=8)
    max_area = 0
    optimal_point = None
    for i in range(1, num_labels):
        _, _, _, _, area = stats[i]
        if area > max_area:
            max_area = area
            optimal_point = tuple(map(int, centroids[i]))
    return optimal_point

def find_boundaries(black_white_image, optimal_point, grid_size=32, threshold=0.1):
    height, width = black_white_image.shape
    optimal_x = int(optimal_point[0])

    def sum_white_pixels(x):
        return np.sum(black_white_image[:, x] == 255)

    white_sums = [sum_white_pixels(x) for x in range(0, width, grid_size)]

    left_boundary = right_boundary = None

    for i in range(optimal_x // grid_size, 0, -1):
        if abs(white_sums[i] - white_sums[i - 1]) > threshold * white_sums[i - 1]:
            left_boundary = i * grid_size
            break

    for i in range(optimal_x // grid_size, len(white_sums) - 1):
        if abs(white_sums[i] - white_sums[i + 1]) > threshold * white_sums[i + 1]:
            right_boundary = (i + 1) * grid_size
            break

    return left_boundary, right_boundary
