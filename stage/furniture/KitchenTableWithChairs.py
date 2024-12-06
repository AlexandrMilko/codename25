from .Furniture import FloorFurniture
from constants import Path
import numpy as np
import cv2


class KitchenTableWithChairs(FloorFurniture):
    def __init__(self, model_path):
        super().__init__(model_path)

    @staticmethod
    def find_placement_pixel(floor_layout_path: str) -> list[tuple[tuple[int, int], float]]:
        image = cv2.imread(floor_layout_path, cv2.IMREAD_GRAYSCALE)

        # Получаем размеры комнаты
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        # Проверяем, находится ли центр в допустимой области
        if KitchenTableWithChairs.square_inside_figure((center[0], center[1], 50), image):  # 50 - пример размера стола
            angle = 0  # Центр комнаты не требует поворота
            return [(center, angle)]
        return []

    @staticmethod
    def find_angle(image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Calculate the sides of the rectangle
        sides = [(box[i], box[(i + 1) % 4]) for i in range(4)]
        side_lengths = [cv2.norm(side[0] - side[1]) for side in sides]

        # Find the largest side
        largest_side = sides[np.argmax(side_lengths)]

        # Calculate the angle of the largest side with respect to the horizontal axis
        dx = largest_side[1][0] - largest_side[0][0]
        dy = largest_side[1][1] - largest_side[0][1]
        angle = np.arctan2(dy, dx) * 180 / np.pi

        return angle

    @staticmethod
    def square_inside_figure(square, shape):
        x, y, size = square
        square_pixels = shape[y:y + size, x:x + size]
        # Assuming the shape is represented by 255 (white)
        inside_count = np.sum(square_pixels == 255)
        total_count = size * size
        acceptable_transcend = 0.9 * total_count
        return inside_count > acceptable_transcend

    @staticmethod
    def crate_grid(image):
        shape_height, shape_width = image.shape[:2]
        square_size = min(shape_height, shape_width) // 3  # Adjusting the size for visibility
        x_coords = np.arange(0, shape_width, square_size)
        y_coords = np.arange(0, shape_height, square_size)

        return x_coords, y_coords, square_size

    @staticmethod
    def rotate_coordinates(x_coords, y_coords, angle, origin):
        angle_rad = np.deg2rad(angle)
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad), np.cos(angle_rad)]])

        rotated_coords = []
        for x in x_coords:
            for y in y_coords:
                rotated = np.dot(rotation_matrix, np.array([x - origin[0], y - origin[1]])) + origin
                rotated_coords.append((int(rotated[0]), int(rotated[1])))
        return rotated_coords

    @staticmethod
    def find_squares(rotated_coords, square_size, image):
        squares = []
        for x, y in rotated_coords:
            if KitchenTableWithChairs.square_inside_figure((x, y, square_size), image):
                squares.append((x, y, square_size))
        return squares

    @staticmethod
    def find_square_center(squares):
        centers = []
        for square in squares:
            x, y, size = square
            center_x = x + size // 2
            center_y = y + size // 2
            centers.append((center_x, center_y))

        return centers