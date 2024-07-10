from .Furniture import HangingFurniture
import numpy as np
import math
import cv2


class Curtain(HangingFurniture):
    default_angles = 0, 0, 90

    def __init__(self, default_height, model_path):
        super().__init__(model_path)
        self.default_height = default_height

    def calculate_height_scale(self, curtains_height):
        return curtains_height / self.default_height

    @staticmethod
    def find_perspective_angle(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        angle_radians = math.atan2(dy, dx)
        angle_degrees = math.degrees(angle_radians)
        if angle_degrees < 0:
            angle_degrees += 360
        return angle_degrees

    @staticmethod
    def find_placement_pixel(window_mask_path: str) -> list[list[tuple[int, int]]]:
        img = cv2.imread(window_mask_path, cv2.IMREAD_GRAYSCALE)
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(thresh, kernel, iterations=1)
        img = cv2.dilate(erosion, kernel, iterations=1)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        final_points = []
        # img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for contour in contours:
            contour_points = contour[:, 0, :]

            # Находим крайние точки
            top_left = contour_points[np.argmin(contour_points[:, 0] + contour_points[:, 1])]
            top_right = contour_points[np.argmax(contour_points[:, 0] - contour_points[:, 1])]

            print("Координаты угловых точек:")
            print("Верхняя левая:", top_left)
            print("Верхняя правая:", top_right)

            angle_radians = math.radians(
                Curtain.find_perspective_angle(top_left[0], top_left[1], top_right[0], top_right[1]))

            # Вычисление координат точек слева и справа от верхних углов
            right_top_point = (
                int(top_left[0] - 20 * math.cos(angle_radians)), -10 + int(top_left[1] - 20 * math.sin(angle_radians)))
            left_top_point = (
                int(top_right[0] + 20 * math.cos(angle_radians)),
                -10 + int(top_right[1] + 20 * math.sin(angle_radians)))
            point = [left_top_point, right_top_point]
            final_points.append(point)

            # cv2.circle(img_vis, right_top_point, 5, (0, 0, 255), -1)  # красная точка - верхняя левая
            # cv2.circle(img_vis, left_top_point, 5, (0, 255, 0), -1)  # зеленая точка - верхняя правая

        # cv2.imshow('Image with Points', img_vis)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return final_points
