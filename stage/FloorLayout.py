import os

import cv2
import numpy as np
import open3d as o3d

from constants import Path, Config
from .LayoutSide import LayoutSide


class FloorLayout:
    def __init__(self, ply_path, middle_points_3d_dict, window_door_borders_3d_dict, output_image_path=Path.FLOOR_LAYOUT_IMAGE.value):

        """
        :param ply_path: path to the .ply file that represents 3d floor plane
        :param output_path: path to save the debug layout image
        :param middle_points_3d_dict: 3d points to be converted into 2d layout pixel coords
        :return: dictionary of converted 2D points and pixels-per-meter ratio
        """

        self.ply_path = ply_path
        self.middle_points_3d_dict = middle_points_3d_dict
        self.window_door_borders_3d_dict = window_door_borders_3d_dict
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
        points_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Add camera and relative point for pixel-per-meter calculation
        self.middle_points_3d_dict['camera'] = [0, 0, 0]
        self.middle_points_3d_dict['point_for_calculating_ratio'] = [0.2, 0.2, 0]

        # Process user's points
        all_points = floor_points.copy()
        for point_name in self.middle_points_3d_dict.keys():
            print(self.middle_points_3d_dict[point_name], " self.middle_points_3d_dict[point_name]")
            middle_point = self.middle_points_3d_dict[point_name]

            # We reverse x-axis, because in blender it points to the opposite than in image pixel coordinate system
            middle_point[0] = -middle_point[0]

            # Append user points to floor points
            all_points = np.vstack([all_points, np.array(middle_point)])

        # Find min and max coordinates of the floor
        min_coords = all_points.min(axis=0)
        max_coords = all_points.max(axis=0)

        # Normalize floor points to image dimensions
        norm_points = (floor_points - min_coords) / (max_coords - min_coords)
        norm_points[:, 0] = norm_points[:, 0] * (width - 1)
        norm_points[:, 1] = norm_points[:, 1] * (height - 1)

        # Visualize all points on the image
        for point in norm_points:
            pixel_x = int(point[0])
            pixel_y = int(point[1])
            cv2.circle(points_image, (pixel_x, pixel_y), 3, (255, 255, 255), -1)  # White color for all points

        # Convert 3D window, doors and camera middle points to 2D pixels on floor layout
        result = dict()
        for point_name in self.middle_points_3d_dict.keys():
            print(self.middle_points_3d_dict)
            middle_point = self.middle_points_3d_dict[point_name]
            print(middle_point)
            x_3d, y_3d, _ = middle_point
            print(f"3D Point: {middle_point}")
            pixel_x = int((x_3d - min_coords[0]) / (max_coords[0] - min_coords[0]) * (width - 1))
            pixel_y = int((y_3d - min_coords[1]) / (max_coords[1] - min_coords[1]) * (height - 1))

            # Ensure pixel coordinates are within bounds
            pixel_x = np.clip(pixel_x, 0, width - 1)
            pixel_y = np.clip(pixel_y, 0, height - 1)
            print(f"Mapped to 2D: x={pixel_x}, y={pixel_y}")  # Debug message
            result[point_name] = [pixel_x, pixel_y]
            # cv2.circle(points_image, (pixel_x, pixel_y), 5, (0, 0, 255), -1)  # Red color for specific points

        # Convert 3D window and doors border points to 2D pixels on floor layout
        self.window_door_borders_pixels = dict()
        for zone_name in self.window_door_borders_3d_dict.keys():
            left_offset, right_offset = self.window_door_borders_3d_dict[zone_name]
            zone_pixel_borders = []
            for offset in left_offset, right_offset:
                x_3d, y_3d, _ = offset
                # We reverse x-axis, because in blender it points to the opposite than in image pixel coordinate system
                x_3d = -x_3d
                pixel_x = int((x_3d - min_coords[0]) / (max_coords[0] - min_coords[0]) * (width - 1))
                pixel_y = int((y_3d - min_coords[1]) / (max_coords[1] - min_coords[1]) * (height - 1))

                # Ensure pixel coordinates are within bounds
                pixel_x = np.clip(pixel_x, 0, width - 1)
                pixel_y = np.clip(pixel_y, 0, height - 1)
                print(f"Mapped to 2D: x={pixel_x}, y={pixel_y}")  # Debug message
                zone_pixel_borders.append([pixel_x, pixel_y])
            self.window_door_borders_pixels[zone_name] = zone_pixel_borders


        if self.output_image_path is not None:
            os.makedirs(os.path.dirname(self.output_image_path), exist_ok=True)
            cv2.imwrite(self.output_image_path, points_image)
            cv2.imwrite(Path.FLOOR_POINTS_IMAGE.value, points_image)

            FloorLayout.clear_floor_layout(self.output_image_path, self.output_image_path)

        self.pixels_dict = result
        camera_pixel = self.pixels_dict['camera']

        refined_image = FloorLayout.refine_contours(self.output_image_path) # We perform additional cleaning made by Vova
        cv2.imwrite(self.output_image_path, refined_image)
        FloorLayout.fill_small_contours(self.output_image_path, self.output_image_path, Config.FLOOR_LAYOUT_CONTOUR_SIZE_TO_REMOVE.value)
        FloorLayout.fill_layout_with_camera(self.output_image_path, camera_pixel, self.output_image_path)

        pixels_per_meter_ratio = self.calculate_pixels_per_meter_ratio()
        print(pixels_per_meter_ratio)

        self.ratio_x, self.ratio_y = pixels_per_meter_ratio

    def calculate_pixels_per_meter_ratio(self):
        """
        offsets: points in 3d space that were converted to the pixels in dictionary format
        pixels: pixel coordinates on floor layout image as a result of conversion in dictionary format
        WARNING! Both dictionaries must have 'camera' and 'point_for_calculating_ratio' keys
        """
        camera_pixel = self.pixels_dict['camera']
        point_pixel = self.pixels_dict['point_for_calculating_ratio']

        camera_offset = self.middle_points_3d_dict['camera']
        point_offset = self.middle_points_3d_dict['point_for_calculating_ratio']
        pixels_x_diff = camera_pixel[0] - point_pixel[0]
        pixels_y_diff = camera_pixel[1] - point_pixel[1]
        offsets_x_diff = camera_offset[0] - point_offset[0]
        offsets_y_diff = camera_offset[1] - point_offset[1]
        ratio_x = pixels_x_diff / offsets_x_diff
        ratio_y = pixels_y_diff / offsets_y_diff
        return ratio_x, ratio_y

    def pixel_to_offset(self):
        pass

    def get_pixels_per_meter_ratio(self):
        return self.ratio_x, self.ratio_y

    def get_middle_points_3d_dict(self):
        return self.middle_points_3d_dict

    def get_pixels_dict(self):
        return self.pixels_dict

    @staticmethod
    def find_all_tangent_zones(point1, point2, exclusion_zones, exclude_distance):
        zone_names = []
        for zone_name, zone_coords in exclusion_zones.items():
            exclusion_point = np.array(zone_coords)

            # Vector from point1 to point2 (the line segment)
            line_vec = np.array(point2) - np.array(point1)

            # Vector from point1 to the exclusion point
            point_vec = exclusion_point - np.array(point1)

            # Project point_vec onto the line_vec to find the closest point on the line
            line_len = np.linalg.norm(line_vec)
            if line_len == 0:
                continue  # Skip degenerate lines (though they shouldn't occur here)

            projection = np.dot(point_vec, line_vec) / (line_len ** 2)
            if projection < 0:
                closest_point = np.array(point1)
            elif projection > 1:
                closest_point = np.array(point2)
            else:
                closest_point = np.array(point1) + projection * line_vec

            # Calculate the distance from the exclusion point to the closest point on the line
            distance_to_line = np.linalg.norm(closest_point - exclusion_point)

            if distance_to_line < exclude_distance:
                zone_names.append(zone_name)

        return zone_names

    @staticmethod
    def is_tangent_to_any(point1, point2, exclusion_zones, exclude_distance, exclude_distance_doorway):
        for zone_name, zone_coords in exclusion_zones.items():
            exclusion_point = np.array(zone_coords)

            # Vector from point1 to point2 (the line segment)
            line_vec = np.array(point2) - np.array(point1)

            # Vector from point1 to the exclusion point
            point_vec = exclusion_point - np.array(point1)

            # Project point_vec onto the line_vec to find the closest point on the line
            line_len = np.linalg.norm(line_vec)
            if line_len == 0:
                continue  # Skip degenerate lines (though they shouldn't occur here)

            projection = np.dot(point_vec, line_vec) / (line_len ** 2)
            if projection < 0:
                closest_point = np.array(point1)
            elif projection > 1:
                closest_point = np.array(point2)
            else:
                closest_point = np.array(point1) + projection * line_vec

            # Calculate the distance from the exclusion point to the closest point on the line
            distance_to_line = np.linalg.norm(closest_point - exclusion_point)

            if zone_name.startswith("doorway"):
                if distance_to_line < exclude_distance_doorway:
                    return True
            else:
                if distance_to_line < exclude_distance:
                    return True

        return False

    @staticmethod
    def draw_points_and_contours(exclusion_zones, exclude_distance, sides, window_door_borders_pixels, exclude_distance_doorway):
        image = cv2.imread(Path.FLOOR_LAYOUT_IMAGE.value)

        for side in sides:
            pt1, pt2 = side.get_points()
            cv2.line(image, pt1, pt2, (255, 0, 0), 2)  # Blue contours

        for name, point in exclusion_zones.items():
            if name.startswith("doorway"):
                # Draw the exclusion circle
                cv2.circle(image, point, exclude_distance_doorway, (0, 255, 0), 2)  # Green circles
            else:
                cv2.circle(image, point, exclude_distance, (0, 255, 0), 2)  # Green circles
            # Draw the point itself
            cv2.circle(image, point, 5, (0, 0, 255), -1)  # Red points

        for zone_name in window_door_borders_pixels.keys():
            left_border_pixel, right_border_pixel = window_door_borders_pixels[zone_name]
            cv2.line(image, left_border_pixel, right_border_pixel, (0, 128, 128), 2)

        cv2.imwrite(Path.FLOOR_LAYOUT_DEBUG_IMAGE.value, image)


    def find_all_sides(self, exclude_distance=50, exclude_length=1.5, exclude_distance_doorway=25):
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
        biggest_contour = max(approx_contours, key=cv2.contourArea)
        for i in range(len(biggest_contour)):
            pt1 = biggest_contour[i][0]
            pt2 = biggest_contour[(i + 1) % len(biggest_contour)][0]
            # Exclude sides that are too close to the camera, windows, or doors
            zone_names = FloorLayout.find_all_tangent_zones(pt1, pt2, exclusion_zones, exclude_distance)
            tangent_borders = [self.window_door_borders_pixels[zone_name] for zone_name in self.window_door_borders_pixels.keys() if zone_name in zone_names]
            free_side_segments = FloorLayout.find_empty_places_on_wall(pt1, pt2, tangent_borders)
            if tangent_borders:
                FloorLayout.visualize_wall_segments(pt1, pt2, tangent_borders, free_side_segments)
            for segment in free_side_segments:
                side = LayoutSide(segment)
                if (not self.is_tangent_to_any(segment[0], segment[1], exclusion_zones, exclude_distance, exclude_distance_doorway)
                        and side.calculate_wall_length(self.ratio_x, self.ratio_y) >= exclude_length):
                    sides.append(side)
                    
        # sides.sort(reverse=True, key=lambda x: x.calculate_wall_length(self.ratio_x, self.ratio_y))
        self.draw_points_and_contours(exclusion_zones, exclude_distance, sides, self.window_door_borders_pixels, exclude_distance_doorway)

        return sides

    @staticmethod
    def find_empty_places_on_wall(wall_start, wall_end, window_door_borders):
        if not window_door_borders: return [[wall_start, wall_end]]
        def find_intersections(wall_start, wall_end, all_points):
            def line_equation(p1, p2):
                a = p2[1] - p1[1]  # y2 - y1
                b = p1[0] - p2[0]  # x1 - x2
                c = a * p1[0] + b * p1[1]  # ax1 + by1
                return a, b, -c

            def perpendicular_line(a, b, c, point):
                px, py = point
                if b == 0:
                    return 0, 1, -py
                elif a == 0:
                    return 1, 0, -px
                else:
                    perp_a = -b
                    perp_b = a
                    perp_c = b * px - a * py
                    return perp_a, perp_b, perp_c

            def line_intersection(a1, b1, c1, a2, b2, c2):
                determinant = a1 * b2 - a2 * b1
                if determinant == 0:
                    return None
                x = (b2 * -c1 - b1 * -c2) / determinant
                y = (a1 * -c2 - a2 * -c1) / determinant
                return x, y

            a1, b1, c1 = line_equation(wall_start, wall_end)
            intersections = []

            for point in all_points:
                perp_a, perp_b, perp_c = perpendicular_line(a1, b1, c1, point)
                intersection = line_intersection(a1, b1, c1, perp_a, perp_b, perp_c)
                if intersection is not None:
                    intersections.append(intersection)

            intersections = sorted(
                intersections, key=lambda p: np.linalg.norm(np.array(p) - np.array(wall_start))
            )

            return intersections

        # Flatten dividing line points into one list
        all_points = [point for line in window_door_borders for point in line]

        # Find intersections
        intersections = find_intersections(wall_start, wall_end, all_points)

        # Create segments from intersections
        line_segments = []
        prev_point = wall_start
        for point in intersections:
            line_segments.append((prev_point, point))
            prev_point = point
        line_segments.append((prev_point, wall_end))

        # Determine if the segment center lies between any projected point pair
        def is_center_in_between(center, point_pairs):
            for start, end in point_pairs:
                # Ensure start and end are in consistent order. Sometimes points are not arranged in the correct order
                x_min, x_max = sorted([start[0], end[0]])
                y_min, y_max = sorted([start[1], end[1]])

                # Check if the center is within the bounds
                if x_min <= center[0] <= x_max and y_min <= center[1] <= y_max:
                    return True
            return False

        # Pair dividing points to check the center
        point_pairs = [(window_door_borders[i][0], window_door_borders[i][1]) for i in range(len(window_door_borders))]

        # Exclude segments whose centers are in between any pair of projected points
        filtered_segments = []
        for segment in line_segments:
            center = (
                (segment[0][0] + segment[1][0]) / 2,
                (segment[0][1] + segment[1][1]) / 2,
            )
            print(center, point_pairs)
            if not is_center_in_between(center, point_pairs):
                filtered_segments.append(segment)

        return filtered_segments

    @staticmethod
    def visualize_wall_segments(wall_start, wall_end, window_door_borders, filtered_segments):
        import matplotlib.pyplot as plt
        # Visualization
        plt.figure(figsize=(8, 8))

        # Plot Line A
        plt.plot(
            [wall_start[0], wall_end[0]],
            [wall_start[1], wall_end[1]],
            label="Line A",
            color="blue",
            linewidth=2,
        )

        # Plot dividing line points
        for line_points in window_door_borders:
            for point in line_points:
                plt.scatter(*point, color="red", label="Dividing Line Points")
            plt.plot(
                [line_points[0][0], line_points[1][0]],
                [line_points[0][1], line_points[1][1]],
                color="green",
                linestyle="dotted",
                label="Dividing Line",
            )

        # Plot filtered segments and their centers
        for segment in filtered_segments:
            # Plot the segment
            plt.plot(
                [segment[0][0], segment[1][0]],
                [segment[0][1], segment[1][1]],
                label="Filtered Segment",
                color="orange",
                linestyle="dashed",
            )
            # Compute and plot the center of the segment
            center = (
                (segment[0][0] + segment[1][0]) / 2,
                (segment[0][1] + segment[1][1]) / 2,
            )
            plt.scatter(*center, color="purple", label="Segment Center", zorder=5)

        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        plt.legend(unique_labels.values(), unique_labels.keys())

        plt.grid(True)
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Filtered Segments of Line A with Multiple Dividing Lines")
        plt.axis("equal")

        from io import BytesIO
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)  # Rewind the buffer to the beginning
        plt.close()  # Close the plt to free resources

        with open(Path.WALL_SEGMENTS_DEBUG_IMAGE.value, "wb") as f:
            f.write(buffer.getvalue())

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
        # Step 1: Load the image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Step 2: Threshold the image to create a binary image
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        # Step 3: Apply morphological closing to close gaps in the border
        kernel = np.ones((15, 15), np.uint8)
        closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

        # Step 4: Invert the image so the background is black (0) and the object is white (255)
        inverted_image = cv2.bitwise_not(closed_image)

        # Step 5: Perform flood fill from a point outside the object
        h, w = inverted_image.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        flood_filled_image = inverted_image.copy()

        # Start filling from point (0, 0) which should be background
        cv2.floodFill(flood_filled_image, mask, (0, 0), (255, 255, 255))

        # Step 6: Invert the flood-filled image back to get the object filled
        filled_image = cv2.bitwise_not(flood_filled_image)

        # Step 7: Save the resulting filled image
        cv2.imwrite(output_path, filled_image)

    @staticmethod
    def fill_small_contours(image_path, output_image_path, area_threshold): # We use it to remove the noise
        # Read the image in grayscale mode
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image at path {image_path} not found.")

        # Threshold the image to ensure binary format (black and white)
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through contours
        for contour in contours:
            # Check the area of the contour
            if cv2.contourArea(contour) < area_threshold:
                # Fill the contour with black
                cv2.drawContours(binary_image, [contour], -1, (0), thickness=cv2.FILLED)

        # Save the processed image
        cv2.imwrite(output_image_path, binary_image)

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
        # To remove the gap between triangle and floor points
        FloorLayout.clear_floor_layout(output_path, output_path)

    @staticmethod
    def refine_contours(image_path):
        img = cv2.imread(image_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_without_noise = np.zeros_like(img)

        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(image_without_noise, [approx], 0, (255, 255, 255), -1)

        return image_without_noise
