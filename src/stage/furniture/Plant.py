from stage.furniture.Furniture import FloorFurniture
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
