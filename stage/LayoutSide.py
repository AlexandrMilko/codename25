import numpy as np
import math


class LayoutSide:
    def __init__(self, points):
        self.points = [list(map(int, point)) for point in points]
        first_point = points[0]
        second_point = points[1]
        self.middle_point = (first_point[0] + second_point[0]) // 2, (first_point[1] + second_point[1]) // 2

    def calculate_wall_angle(self):
        top_point = (self.middle_point[0], 0)

        vector_top_to_middle = np.array(self.middle_point) - np.array(top_point)

        vector_wall = np.array(self.points[1]) - np.array(self.points[0])
        perpendicular_vector = np.array([-vector_wall[1], vector_wall[0]])

        angle_radians = math.atan2(
            np.linalg.det([vector_top_to_middle, perpendicular_vector]),
            np.dot(vector_top_to_middle, perpendicular_vector)
        )
        angle_degrees = math.degrees(angle_radians)

        return -angle_degrees

    def calculate_wall_length(self, ratio_x, ratio_y):
        first_leg = abs(self.points[0][0] - self.points[1][0]) / ratio_x
        second_leg = abs(self.points[0][1] - self.points[1][1]) / ratio_y
        return math.sqrt(first_leg ** 2 + second_leg ** 2)

    def calculate_wall_height(self, ratio_y):
        return abs(self.points[0][1] - self.points[1][1]) / ratio_y

    def get_points(self):
        return self.points

    def get_middle_point(self):
        return self.middle_point

    def __repr__(self):
        return f"Floor Layout Side: {self.points}, {self.middle_point}"
