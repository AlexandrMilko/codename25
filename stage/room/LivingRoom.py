from postprocessing.postProcessing import PostProcessor
from preprocessing.preProcessSegment import ImageSegmentor
from constants import Path
from tools import resize_and_save_image
from .Room import Room
from .. import Floor
from ..furniture.Furniture import Furniture
import os
import numpy as np
import cv2
from constants import Config

class LivingRoom(Room):
    def stage(self):
        camera_height, pitch_rad, roll_rad, height, scene_render_parameters = self.prepare_empty_room_data()

        area = self.floor_layout.estimate_area_from_floor_layout()
        print(area, "AREA in m2")

        all_sides = self.floor_layout.find_all_sides_sorted_by_length()
        print(all_sides, "ALL SIDES")

        # Add living room sofa, table, and sofachairs
        living_room_set_parameters = self.calculate_sofa_with_table_parameters(all_sides, (pitch_rad, roll_rad))

        scene_render_parameters['objects'] = [
            # *curtains_parameters,
            living_room_set_parameters
        ]
        # After our parameters calculation som of them will be equal to None, we have to remove them
        scene_render_parameters['objects'] = [item for item in scene_render_parameters['objects'] if item is not None]

        import json
        print(json.dumps(scene_render_parameters, indent=4))

        Furniture.start_blender_render(scene_render_parameters)

        if Config.DO_POSTPROCESSING.value and Config.UI.value == "comfyui":
            processor = PostProcessor()
            processor.execute()

    def calculate_sofa_with_table_parameters(self, all_sides, camera_angles_rad: tuple):
        if len(all_sides) > 0:
            side = all_sides.pop(0)
        else:
            return None

        from stage.furniture.LivingRoomSet import LivingRoomSet
        pitch_rad, roll_rad = camera_angles_rad
        biggest_side = side

        ratio_x, ratio_y = self.floor_layout.get_pixels_per_meter_ratio()
        pixels_dict = self.floor_layout.get_pixels_dict()

        # TODO rename to offset_pixel and use same name in all the rest of the functions
        offset_point, yaw_angle = self.calculate_angle_and_offset_pixel(biggest_side)
        pixel_diff = -1 * (offset_point[0] - pixels_dict['camera'][0]), offset_point[1] - pixels_dict['camera'][1]
        living_room_set_offset = self.floor_layout.calculate_offset_from_pixel_diff(pixel_diff, (ratio_x, ratio_y))

        living_room_set = LivingRoomSet()
        render_parameters = (
            living_room_set.calculate_rendering_parameters(self, living_room_set_offset, yaw_angle, (roll_rad, pitch_rad)))
        return render_parameters

    def calculate_angle_and_offset_pixel(self, longest_side):
        # Load the original image
        image_path = Path.FLOOR_LAYOUT_IMAGE.value
        new_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Find the largest rectangle that is perpendicular to this blue side
        best_rect, best_angle, best_center = find_largest_rectangle_perpendicular_to_side(new_image, longest_side)

        x1, y1, x2, y2 = best_rect
        cv2.rectangle(new_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(new_image, best_center, 3, (0, 0, 255))

        # Save and show the final result for the largest rectangle
        output_path_rectangle = Path.FLOOR_LAYOUT_DEBUG_IMAGE.value
        cv2.imwrite(output_path_rectangle, new_image)

        # Display the largest rectangle image
        # plt.figure()
        # plt.imshow(cv2.cvtColor(final_result_image, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()

        return best_center, best_angle

def find_largest_inner_rectangle(mask):
    rows, cols = mask.shape
    max_area = 0
    best_rect = (0, 0, 0, 0)

    heights = np.zeros((rows, cols), dtype=int)

    for row in range(rows):
        for col in range(cols):
            if mask[row, col] == 255:
                heights[row, col] = heights[row - 1, col] + 1 if row > 0 else 1

    for row in range(rows):
        stack = []
        for col in range(cols + 1):
            current_height = heights[row, col] if col < cols else 0
            while stack and current_height < heights[row, stack[-1]]:
                h = heights[row, stack.pop()]
                w = col if not stack else col - stack[-1] - 1
                area = h * w
                if area > max_area:
                    max_area = area
                    best_rect = (stack[-1] + 1 if stack else 0, row - h + 1, w, h)
            stack.append(col)

    return best_rect

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR)
    return rotated_image, matrix

def inverse_rotate(image, matrix):
    (h, w) = image.shape[:2]
    inverse_matrix = cv2.invertAffineTransform(matrix)
    rotated_back_image = cv2.warpAffine(image, inverse_matrix, (w, h), flags=cv2.INTER_LINEAR)
    return rotated_back_image

def add_extra_padding(image, padding_size=150):
    return cv2.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=0)

def dynamic_remove_padding(image, original_image_shape):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return image[y:y + h, x:x + w]
    else:
        return image

def calculate_perpendicular_angle(side):
    angle = side.calculate_wall_angle()  # Get the angle of the side in degrees
    perpendicular_angle = (angle + 90) % 360  # Find the perpendicular angle by adding 90 degrees
    return perpendicular_angle

def invert_rotate_pixel(pixel, matrix):
    # Rotate best_center back to the original orientation
    best_center_homogeneous = np.array([pixel[0], pixel[1], 1])  # Convert to homogeneous coordinates
    inverse_matrix = cv2.invertAffineTransform(matrix)
    original_best_center = inverse_matrix @ best_center_homogeneous  # Apply inverse rotation

    # Convert the result back to integer coordinates
    original_best_center = (int(original_best_center[0]), int(original_best_center[1]))
    return original_best_center

def find_largest_rectangle_perpendicular_to_side(image, side):
    # Calculate the perpendicular angle to the given side
    perpendicular_angle = calculate_perpendicular_angle(side)

    # Rotate the image so that the side is perpendicular
    rotated_image, matrix = rotate_image(image, perpendicular_angle)

    # Find the largest rectangle in the rotated image
    x, y, w, h = find_largest_inner_rectangle(rotated_image)

    rotated_best_rect = (x, y, w, h)
    best_angle = perpendicular_angle % 90

    centroid_x = x + w // 2
    centroid_y = y + h // 2
    rotated_best_center = (centroid_x, centroid_y)

    original_best_center = invert_rotate_pixel(rotated_best_center, matrix)
    original_best_rectangle = *invert_rotate_pixel([x,y], matrix), *invert_rotate_pixel([x+w, y+h], matrix)

    return original_best_rectangle, best_angle, original_best_center

def is_side_too_close_to_exclusion_zone(pt1, pt2, exclusion_zones, exclude_distance):
    pt1_3d = np.array([pt1[0], pt1[1], 0])  # Convert to 3D by adding a 0 for z-axis
    pt2_3d = np.array([pt2[0], pt2[1], 0])  # Convert to 3D by adding a 0 for z-axis

    for key, point in exclusion_zones.items():
        point_3d = np.array([point[0], point[1], 0])  # Convert exclusion point to 3D

        # Calculate the distance from exclusion point to the line (pt1, pt2)
        dist = np.abs(np.cross(pt2_3d - pt1_3d, pt1_3d - point_3d)) / np.linalg.norm(pt2_3d - pt1_3d)

        # The cross product result will have a z-component only, so we take the magnitude of the z-axis
        if dist[2] < exclude_distance:
            return True
    return False

