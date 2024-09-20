from .Furniture import FloorFurniture
from constants import Path
import numpy as np
import cv2


class Plant(FloorFurniture):
    def __init__(self, model_path=Path.PLANT_MODEL.value):
        super().__init__(model_path)

    @staticmethod
    def is_near_border(x, y, margin, width, height):
        return x < margin or x > width - margin or y < margin or y > height - margin

    @staticmethod
    def find_placement_pixel(floor_mask_path: str) -> list[list[int, int]]:
        # Load the image
        img = cv2.imread(floor_mask_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # WARNING useful only when shape has gray borders
        # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

        # Apply erosion to move points inside
        erosion = 40
        kernel = np.ones((erosion, erosion), np.uint8)  # Adjust the kernel size as needed
        eroded = cv2.erode(gray, kernel, iterations=1)

        # Find contours               (thresh,..
        contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]

        # Approximate the contour
        contour_precision = 0.01
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, contour_precision * perimeter, True)

        # Set margin from border
        margin = 10 + erosion
        # Get image dimensions
        height, width = img.shape[:2]

        points = []
        # Draw points and contours
        for point in approx:
            x, y = point[0]
            if not Plant.is_near_border(x, y, margin, width, height):
                points.append(point[0])
                # cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

        # cv2.drawContours(img, [approx], -1, (0, 255, 0))

        # cv2.imshow('Contour Approximation', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return points

    @staticmethod
    def find_floor_layout_placement_pixels(floor_layout_image_path, min_distance_between_points=100, offset=25):
        image = cv2.imread(floor_layout_image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        approx_contours = []
        for cnt in contours:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            approx_contours.append(approx)

        selected_points = []
        image_height, image_width = image.shape[:2]
        lower_limit = int(image_height * 0.6)

        center_x = image_width // 2
        center_y = image_height // 2

        for cnt in approx_contours:
            for point in cnt:
                point_position = point[0]

                if point_position[1] < lower_limit:
                    continue

                point_position[0] = min(max(point_position[0], offset), image_width - offset)
                point_position[1] = min(max(point_position[1], offset), image_height - offset)

                direction_x = center_x - point_position[0]
                direction_y = center_y - point_position[1]

                magnitude = np.sqrt(direction_x ** 2 + direction_y ** 2)
                if magnitude > 0:
                    direction_x /= magnitude
                    direction_y /= magnitude
                    point_position[0] += direction_x * offset
                    point_position[1] += direction_y * offset

                if selected_points:
                    distances = np.linalg.norm(np.array(selected_points) - point_position, axis=1)
                    if np.all(distances >= min_distance_between_points):
                        selected_points.append(point_position)
                else:
                    selected_points.append(point_position)

        return selected_points