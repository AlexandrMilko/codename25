from constants import Path
import open3d as o3d
import numpy as np
import os
import cv2
import math
from .LayoutSide import LayoutSide


class FloorLayout:
    def __init__(self, ply_path, points_dict, output_image_path=Path.FLOOR_LAYOUT_IMAGE.value):

        """
        :param ply_path: path to the .ply file that represents 3d floor plane
        :param output_path: path to save the debug layout image
        :param points_dict: 3d points to be converted into 2d layout pixel coords
        :return: dictionary of converted 2D points and pixels-per-meter ratio
        """

        self.ply_path = ply_path
        self.points_dict = points_dict
        self.output_image_path = output_image_path

        # Load the point cloud
        pcd = o3d.io.read_point_cloud(self.ply_path)
        points = np.asarray(pcd.points)

        # WARNING: Ensure path only contains floor points
        floor_points = points

        # We reverse x-axis, because in blender it points to the opposite than in image pixel coordinate system
        floor_points[:, 0] = -floor_points[:, 0]

        # Initialize layout image
        height, width = 1024, 1024
        layout_image = np.zeros((height, width, 3), dtype=np.uint8)
        points_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Add camera and relative point for pixel-per-meter calculation
        self.points_dict['camera'] = [[0, 0, 0], [0, 0, 0]]
        self.points_dict['point_for_calculating_ratio'] = [[0.2, 0.2, 0], [0.2, 0.2, 0]]

        # Process user's points
        for point_name in self.points_dict.keys():
            print(self.points_dict[point_name], " self.points_dict[point_name]")
            left = self.points_dict[point_name][0]
            right = self.points_dict[point_name][1]

            # We reverse x-axis, because in blender it points to the opposite than in image pixel coordinate system
            left[0] = -left[0]
            right[0] = -right[0]

            # Append user points to floor points
            floor_points = np.vstack([floor_points, np.array(left)])
            floor_points = np.vstack([floor_points, np.array(right)])

        # Find min and max coordinates of the floor
        min_coords = floor_points.min(axis=0)
        max_coords = floor_points.max(axis=0)

        # Print min and max coordinates for debugging
        print("Min coordinates:", min_coords)
        print("Max coordinates:", max_coords)

        # Normalize floor points to image dimensions
        norm_points = (floor_points - min_coords) / (max_coords - min_coords)
        norm_points[:, 0] = norm_points[:, 0] * (width - 1)
        norm_points[:, 1] = norm_points[:, 1] * (height - 1)

        for point in norm_points:
            pixel_x = int(point[0])
            pixel_y = int(point[1])
            cv2.circle(layout_image, (pixel_x, pixel_y), 5, (255, 255, 255), -1)

        # Print normalized points for debugging
        print("Normalized points (first 5):", norm_points[:5])

        # Visualize all points on the image
        for point in norm_points:
            pixel_x = int(point[0])
            pixel_y = int(point[1])
            cv2.circle(points_image, (pixel_x, pixel_y), 1, (255, 255, 255), -1)  # White color for all points

        # Convert 3D points to 2D pixels
        result = dict()
        for point_name in self.points_dict.keys():
            print(self.points_dict)
            left = self.points_dict[point_name][0]
            right = self.points_dict[point_name][1]
            result[point_name] = []
            for point in left, right:
                x_3d, y_3d, _ = point
                print(f"3D Point: {point}")
                pixel_x = int((x_3d - min_coords[0]) / (max_coords[0] - min_coords[0]) * (width - 1))
                pixel_y = int((y_3d - min_coords[1]) / (max_coords[1] - min_coords[1]) * (height - 1))

                # Ensure pixel coordinates are within bounds
                pixel_x = np.clip(pixel_x, 0, width - 1)
                pixel_y = np.clip(pixel_y, 0, height - 1)
                print(f"Mapped to 2D: x={pixel_x}, y={pixel_y}")  # Debug message
                result[point_name].append([pixel_x, pixel_y])
                # cv2.circle(layout_image, (pixel_x, pixel_y), 5, (0, 0, 255), -1)  # Red color for specific points

        if self.output_image_path is not None:
            os.makedirs(os.path.dirname(self.output_image_path), exist_ok=True)
            cv2.imwrite(self.output_image_path, layout_image)
            FloorLayout.clear_floor_layout(self.output_image_path, self.output_image_path)
            cv2.imwrite(Path.POINTS_DEBUG_IMAGE.value, points_image)

        self.pixels_dict = result
        left_camera_pixel = self.pixels_dict['camera'][0]
        FloorLayout.fill_layout_with_camera(self.output_image_path, left_camera_pixel, self.output_image_path)
        pixels_per_meter_ratio = self.calculate_pixels_per_meter_ratio()
        print(pixels_per_meter_ratio)

        self.ratio_x, self.ratio_y = pixels_per_meter_ratio

    def calculate_pixels_per_meter_ratio(self):
        """
        offsets: points in 3d space that were converted to the pixels in dictionary format
        pixels: pixel coordinates on floor layout image as a result of conversion in dictionary format
        WARNING! Both dictionaries must have 'camera' and 'point_for_calculating_ratio' keys
        """
        left_camera_pixel = self.pixels_dict['camera'][0]
        left_point_pixel = self.pixels_dict['point_for_calculating_ratio'][0]

        left_camera_offset = self.points_dict['camera'][0]
        left_point_offset = self.points_dict['point_for_calculating_ratio'][0]
        pixels_x_diff = left_camera_pixel[0] - left_point_pixel[0]
        pixels_y_diff = left_camera_pixel[1] - left_point_pixel[1]
        offsets_x_diff = left_camera_offset[0] - left_point_offset[0]
        offsets_y_diff = left_camera_offset[1] - left_point_offset[1]
        ratio_x = pixels_x_diff / offsets_x_diff
        ratio_y = pixels_y_diff / offsets_y_diff
        return ratio_x, ratio_y

    def pixel_to_offset(self):
        pass

    def get_pixels_per_meter_ratio(self):
        return self.ratio_x, self.ratio_y

    def get_points_dict(self):
        return self.points_dict

    def get_pixels_dict(self):
        return self.pixels_dict

    @staticmethod
    def is_tangent_to_any(point1, point2, exclusion_zones, exclude_distance):
        for zone in exclusion_zones.values():
            for exclusion_point in zone:
                if (np.linalg.norm(point1 - exclusion_point) < exclude_distance or
                        np.linalg.norm(point2 - exclusion_point) < exclude_distance):
                    return True
        return False

    def find_all_sides_sorted_by_length(self, exclude_distance=50, exclude_length=2):
        exclusion_zones = self.pixels_dict
        image = cv2.imread(self.output_image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        approx_contours = []
        for cnt in contours:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            approx_contours.append(approx)

        sides = []
        for contour in approx_contours:
            for i in range(len(contour)):
                pt1 = contour[i][0]
                pt2 = contour[(i + 1) % len(contour)][0]
                print(exclusion_zones)
                # Exclude sides that are too close to the camera, windows, or doors
                side = LayoutSide((pt1, pt2))
                if self.is_tangent_to_any(pt1, pt2, exclusion_zones, exclude_distance) or side.calculate_wall_length(self.ratio_x, self.ratio_y) < exclude_length:
                    continue

                sides.append(side)

        sides.sort(reverse=True, key=lambda x: x.calculate_length())

        return sides

    @staticmethod
    def calculate_offset_from_pixel_diff(pixels_diff, ratio):
        """
        pixels_diff: has the following format [pixels_x_diff, pixels_y_diff].
        Represents a difference in pixel coordinates between 2 points

        ratio: has the following format [ratio_x, ratio_y]
        Represents pixels_per_meter_ratio for a floor layout. For both axes.
        """
        pixel_x_diff, pixel_y_diff = pixels_diff
        ratio_x, ratio_y = ratio
        offset_x, offset_y = pixel_x_diff / ratio_x, pixel_y_diff / ratio_y
        return offset_x, offset_y

    def estimate_area_from_floor_layout(self):
        # Загрузка изображения планировки
        layout_image = cv2.imread(self.output_image_path, cv2.IMREAD_GRAYSCALE)

        # Изменение размера изображения так, чтобы ratio_x стало равно ratio_y
        height, width = layout_image.shape
        new_height = int(height * (self.ratio_y / self.ratio_x))
        resized_layout_image = cv2.resize(layout_image, (width, new_height), interpolation=cv2.INTER_NEAREST)

        # Подсчет количества белых пикселей
        white_pixel_count = np.sum(resized_layout_image == 255)

        # Перевод количества белых пикселей в квадратные метры
        meter_area = white_pixel_count / (self.ratio_x * self.ratio_y)

        return meter_area

    @staticmethod
    def clear_floor_layout(image_path, output_path):
        # Removes lonely points and makes the edges smoother for floor layout

        # Create an empty black image
        layout_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply a threshold to ensure the image is binary
        _, binary_image = cv2.threshold(layout_image, 127, 255, cv2.THRESH_BINARY)

        # Define a kernel for morphological operations
        kernel = np.ones((10, 10), np.uint8)

        # Apply morphological opening to remove noise (small dots)
        opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

        # Apply morphological closing to smooth the edges and close small holes
        closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)

        # Find contours in the processed image
        contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Fill the contours on the original layout image (using the processed image as a mask)
        final_image = np.zeros_like(layout_image)
        cv2.fillPoly(final_image, contours, (255, 255, 255))

        # Display the final processed image
        # cv2.imshow("Final Layout Image", final_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite(output_path, final_image)

    @staticmethod
    def fill_layout_with_camera(image_path, camera_pixel, output_path):
        """
        Finds the two topmost corners in a binary image, visualizes them, and draws a triangle using a given third point.

        Parameters:
        - image_path: str, path to the input binary image.
        - third_point: tuple, the (x, y) coordinate of the third point for the triangle.
        - output_path: str, path to save the image with the drawn triangle.

        Returns:
        - top_corners: list of tuples, the (x, y) coordinates of the two topmost corners.
        """
        # Load the binary image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to color image for visualization

        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # List to store potential topmost corners
        potential_corners = []

        # Iterate over contours to find the topmost points
        for contour in contours:
            # Approximate the contour to reduce the number of points
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            for point in approx:
                x, y = point[0]
                potential_corners.append((x, y))

        # Sort potential corners by y-coordinate (ascending)
        sorted_corners = sorted(potential_corners, key=lambda pt: pt[1])

        # Get the two topmost points (having smallest y-coordinates)
        top_corners = sorted_corners[:2]

        # Visualize the two topmost corners by drawing circles
        # for corner in top_corners:
        #     cv2.circle(color_image, corner, 5, (0, 0, 255), -1)  # Red circles for topmost corners

        # Visualize the third point as well
        # cv2.circle(color_image, third_point, 5, (0, 255, 0), -1)  # Green circle for the third point

        # Draw a triangle using the two topmost corners and the given third point
        triangle_points = np.array([top_corners[0], top_corners[1], camera_pixel], np.int32)
        triangle_points = triangle_points.reshape((-1, 1, 2))

        # Fill the triangle on the image
        cv2.fillPoly(color_image, [triangle_points], (255, 255, 255))

        # Save the result
        cv2.imwrite(output_path, color_image)