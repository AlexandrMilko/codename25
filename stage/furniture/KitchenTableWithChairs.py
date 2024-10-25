from .Furniture import FloorFurniture
from constants import Path
import numpy as np
import cv2


class KitchenTableWithChairs(FloorFurniture):
    def __init__(self, model_path=Path.KITCHEN_TABLE_WITH_CHAIRS_MODEL.value):
        super().__init__(model_path)

    @staticmethod
    def find_placement_pixel(floor_layout_path: str, kitchen_mask_path: str) -> list[tuple[tuple[int, int], float]]:
        image = cv2.imread(floor_layout_path, cv2.IMREAD_GRAYSCALE)
        kitchen_mask = cv2.imread(kitchen_mask_path, cv2.IMREAD_GRAYSCALE)  # Маска кухни

        origin = (image.shape[1] // 2, image.shape[0] // 2)  # Центр комнаты
        angle = KitchenTableWithChairs.find_angle(image)

        x_coords, y_coords, square_size = KitchenTableWithChairs.crate_grid(image)
        rotated_coords = KitchenTableWithChairs.rotate_coordinates(x_coords, y_coords, angle, origin)

        squares = KitchenTableWithChairs.find_squares(rotated_coords, square_size, image, kitchen_mask)
        centers = KitchenTableWithChairs.find_square_center(squares)

        # Если нашлись центры, возвращаем их
        if centers:
            return [(center, angle) for center in centers]

        # Если нет доступных центров, возвращаем пиксели ближе к центру
        closest_to_center = min(squares, key=lambda c: cv2.norm(np.array(c[:2]) - np.array(origin)))
        return [(closest_to_center[:2], angle)]

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
        if x < 0 or y < 0 or x + size > shape.shape[1] or y + size > shape.shape[0]:
            return False  # Проверяем, не выходит ли квадрат за пределы изображения
        square_pixels = shape[y:y + size, x:x + size]
        # Проверяем, что квадрат находится внутри белого пространства (предположим, что комната - белая зона)
        inside_count = np.sum(square_pixels == 255)
        total_count = size * size
        acceptable_transcend = 0.9 * total_count  # Допускаем до 10% "прозрачности" пересечения
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
    def find_squares(rotated_coords, square_size, image, kitchen_mask):
        squares = []
        for x, y in rotated_coords:
            if KitchenTableWithChairs.square_inside_figure((x, y, square_size), image) and \
               not KitchenTableWithChairs.square_inside_figure((x, y, square_size), kitchen_mask):
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
