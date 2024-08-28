import numpy as np
import math


class LayoutSide:
    def __init__(self, points):
        self.points = points
        self.middle_point = (points[0] + points[1]) // 2

    def calculate_wall_angle(self):
        # Define the vertical line
        top_point = (self.middle_point[0], 0)

        # Calculate the vector from the top point to the middle point
        vector_top_to_middle = np.array(self.middle_point) - np.array(top_point)

        # Calculate the vector perpendicular to the longest side
        vector_longest_side = np.array(self.points[1]) - np.array(self.points[0])
        perpendicular_vector = np.array([-vector_longest_side[1], vector_longest_side[0]])

        # Calculate the angle between the vertical line and perpendicular one
        angle_radians = math.atan2(
            np.linalg.det([vector_top_to_middle, perpendicular_vector]),
            np.dot(vector_top_to_middle, perpendicular_vector)
        )
        angle_degrees = math.degrees(angle_radians)

        return -angle_degrees  # We return with minus to make it a Blender angle

    def get_middle_point(self):
        return self.middle_point

    def calculate_length(self):
        return np.linalg.norm(self.points[0] - self.points[1])

    def calculate_wall_length(self, ratio_x, ratio_y):
        first_cathetus = abs(self.points[0][0] - self.points[1][0]) / ratio_x
        second_cathetus = abs(self.points[0][1] - self.points[1][1]) / ratio_y
        return math.sqrt(first_cathetus ** 2 + second_cathetus ** 2)

    def __repr__(self):
        return f"Floor Layout Side: {self.points}, {self.middle_point}, length: {self.calculate_length()}"
