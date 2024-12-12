import math

import cv2
import numpy as np

from constants import Path, Config
from preprocessing.preProcessSegment import ImageSegmentor
from stage import Floor
from tools import (get_image_size, calculate_angle_from_top_view, resize_and_save_image,
                   downscale_image_if_bigger, run_subprocess, segment_lang_sam, substract_bbox_from_mask)
from ..FloorLayout import FloorLayout


class Room:
    # BGR, used in segmented images
    window_color = (230, 230, 230)
    door_color = (51, 255, 8)
    floor_color = (50, 50, 80)
    blind_color = (255, 61, 0)  # blind that is set on windows, kinda curtains
    floor_layout = None

    def __init__(self, empty_room_image_path):  # Original image path is an empty space image
        self.empty_room_image_path = empty_room_image_path
        self.focal_length_px = 0

    @staticmethod
    def find_roll_pitch() -> tuple[float, float]:
        from tools import calculate_roll_angle, calculate_pitch_angle, calculate_plane_normal
        plane_normal = calculate_plane_normal(Path.FLOOR_PLY.value)
        roll_rad = math.radians(calculate_roll_angle(plane_normal))
        pitch_rad = math.radians(calculate_pitch_angle(plane_normal))
        return roll_rad, pitch_rad

    def infer_3d(self, pixel: tuple[int, int], pitch_rad: float, roll_rad: float):
        # Using depth-pro methods
        from ml_depth_pro.pro_depth_estimation import image_pixel_to_3d, rotate_3d_point
        print(self.empty_room_image_path, pixel, "IMAGE PATH and PIXEL")
        target_point = image_pixel_to_3d(*pixel, self.empty_room_image_path, self.focal_length_px)
        # We rotate it back to compensate our camera rotation
        offset_relative_to_camera = rotate_3d_point(target_point, -pitch_rad, -roll_rad)
        return offset_relative_to_camera.tolist()  # We convert it to list to avoid serializing errors for blender_script

    def create_floor_layout(self, pitch_rad: float, roll_rad: float):
        horizontal_borders = self.find_horizontal_borders()
        for i, doorway in enumerate(self.doorways_bottom_pixels):
            horizontal_borders[f"doorway_{i}"] = doorway
        print(horizontal_borders)

        middle_points_in_3d = {}
        borders_in_3d = {}
        for name, value in horizontal_borders.items():
            left_pixel, right_pixel = value
            left_offset = self.infer_3d(left_pixel, pitch_rad, roll_rad)
            right_offset = self.infer_3d(right_pixel, pitch_rad, roll_rad)
            middle_offset = [(left_offset[0] + right_offset[0]) / 2,
                             (left_offset[1] + right_offset[1]) / 2,
                             (left_offset[2] + right_offset[2]) / 2]
            middle_points_in_3d[name] = middle_offset
            borders_in_3d[name] = [left_offset, right_offset]

        print(middle_points_in_3d)
        self.floor_layout = FloorLayout(Path.FLOOR_PLY.value, middle_points_in_3d, borders_in_3d)

    def estimate_camera_height(self, camera_angles: tuple[float, float]):
        pitch, roll = camera_angles
        from ml_depth_pro.pro_depth_estimation import rotate_3d_point, image_pixel_to_3d
        import stage.Floor
        floor_pixel = stage.Floor.find_centroid(Path.SEG_INPUT_IMAGE.value)
        point_3d = image_pixel_to_3d(*floor_pixel, self.empty_room_image_path, self.focal_length_px)
        print(f"Floor Centroid: {floor_pixel} -> {point_3d}")
        rotated_point = rotate_3d_point(point_3d, -pitch, -roll)
        z_coordinate = rotated_point[2]
        return abs(z_coordinate)

    @staticmethod
    def find_number_of_windows(windows_mask_path: str) -> int:
        img = cv2.imread(windows_mask_path, cv2.IMREAD_GRAYSCALE)
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(thresh, kernel, iterations=1)
        img = cv2.dilate(erosion, kernel, iterations=1)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Визуализация результатов
        img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Переводим в цвет для визуализации контуров
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

        # Подсчет количества окон
        number_of_windows = len(contours)

        return number_of_windows

    @staticmethod
    def save_windows_mask(segmented_image_path: str, output_windows_mask_path: str):
        image = cv2.imread(segmented_image_path)
        window_rgb_values = Room.window_color
        blind_rgb_values = Room.blind_color

        tolerance = 3
        window_lower_color = np.array([x - tolerance for x in window_rgb_values])
        window_upper_color = np.array([x + tolerance for x in window_rgb_values])
        blind_lower_color = np.array([x - tolerance for x in blind_rgb_values])
        blind_upper_color = np.array([x + tolerance for x in blind_rgb_values])

        window_color_mask = cv2.inRange(image, window_lower_color, window_upper_color)
        blind_color_mask = cv2.inRange(image, blind_lower_color, blind_upper_color)
        combined_mask = cv2.bitwise_or(window_color_mask, blind_color_mask)

        # Debugging: Check if any window pixels were detected
        window_pixel_count = np.count_nonzero(window_color_mask)
        print(f"Window pixels detected: {window_pixel_count}")

        if window_pixel_count == 0:
            print("No window pixels detected. Check the color values and tolerance.")

        # Process the mask further
        _, thresh = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)
        width, height = get_image_size(segmented_image_path)
        kernel = np.ones((height // 25, height // 25), np.uint8)
        erosion = cv2.erode(thresh, kernel, iterations=1)
        color_mask = cv2.dilate(erosion, kernel, iterations=1)

        bw_mask = np.zeros_like(color_mask)
        bw_mask[color_mask != 0] = 255
        cv2.imwrite(output_windows_mask_path, bw_mask)

        # Debugging: Print confirmation of mask creation
        print(f"Window mask saved to {output_windows_mask_path}")

    @staticmethod
    def move_to_target_color(pixel, img, direction):
        target_color = (120, 120, 120)
        x, y = pixel

        if direction == 'left':
            while x > 0 and not np.array_equal(img[y, x], target_color):
                x -= 1
        elif direction == 'right':
            while x < img.shape[1] - 1 and not np.array_equal(img[y, x], target_color):
                x += 1

        return x, y

    def find_horizontal_borders(self):
        target_colors = {
            "door": np.array([8, 255, 51]),  # #08FF33
            "window": np.array([230, 230, 230]),  # #E6E6E6
        }

        object_counter = {
            "door": 0,
            "window": 0
        }

        min_area_threshold = 800

        image = cv2.imread(Path.SEG_INPUT_IMAGE.value)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bottom_pixels = {}

        for object_name, color_value in target_colors.items():
            mask = cv2.inRange(image_rgb, color_value, color_value)
            cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > min_area_threshold:
                    leftmost_bottom_pixel = tuple(contour[contour[:, :, 0].argmin()][0])
                    rightmost_bottom_pixel = tuple(contour[contour[:, :, 0].argmax()][0])

                    leftmost_bottom_pixel_adjusted = self.move_to_target_color(leftmost_bottom_pixel, image_rgb,
                                                                               direction='left')
                    rightmost_bottom_pixel_adjusted = self.move_to_target_color(rightmost_bottom_pixel, image_rgb,
                                                                                direction='right')

                    bottom_pixels[object_name + str(object_counter[object_name])] = (
                        leftmost_bottom_pixel_adjusted, rightmost_bottom_pixel_adjusted
                    )

                    object_counter[object_name] += 1

        return bottom_pixels

    def calculate_painting_parameters(self, camera_angles_rad: tuple):
        from ..furniture.Painting import Painting
        pitch_rad, roll_rad = camera_angles_rad
        painting = Painting()
        left, center, right = painting.find_placement_pixel(Path.SEG_RENDER_IMAGE.value)
        if not all([left, center, right]): raise Exception("No place for painting found")
        left_offset, right_offset = [self.infer_3d((x, center[1]), pitch_rad, roll_rad) for x in (left, right)]
        yaw_angle = calculate_angle_from_top_view(left_offset, right_offset)
        render_parameters = painting.calculate_rendering_parameters(self, center, yaw_angle,
                                                                    (roll_rad, pitch_rad))
        return render_parameters

    def prepare_empty_room_data(self):
        downscale_image_if_bigger(self.empty_room_image_path, self.empty_room_image_path, Config.IMAGE_HEIGHT_LIMIT.value)
        # Image.open(self.empty_room_image_path).save(Path.RENDER_IMAGE.value)
        from ml_depth_pro.pro_depth_estimation import (image_pixels_to_space_and_floor_point_clouds,
                                                       rotate_ply_file_with_colors)

        width, height = get_image_size(self.empty_room_image_path)
        PREPROCESSOR_RESOLUTION_LIMIT = Config.CONTROLNET_HEIGHT_LIMIT.value if height > Config.CONTROLNET_HEIGHT_LIMIT.value else height

        segment = ImageSegmentor(self.empty_room_image_path, Path.SEG_INPUT_IMAGE.value, PREPROCESSOR_RESOLUTION_LIMIT)
        segment.execute()

        resize_and_save_image(Path.SEG_INPUT_IMAGE.value, Path.SEG_INPUT_IMAGE.value, height)
        Floor.save_mask(Path.SEG_INPUT_IMAGE.value, Path.FLOOR_MASK_IMAGE.value)

        doorways_bounding_boxes = segment_lang_sam(Path.INPUT_IMAGE.value, Path.DOOR_SEG_IMG_OUTPUT.value)
        self.doorways_bottom_pixels = substract_bbox_from_mask(Path.FLOOR_MASK_IMAGE.value,
                                                          Path.FLOOR_MASK_IMAGE.value,
                                                          doorways_bounding_boxes)

        Room.save_windows_mask(Path.SEG_INPUT_IMAGE.value, Path.WINDOWS_MASK_IMAGE.value)

        self.focal_length_px = image_pixels_to_space_and_floor_point_clouds(self.empty_room_image_path)
        roll_rad, pitch_rad = np.negative(self.find_roll_pitch())

        rotate_ply_file_with_colors(Path.FLOOR_PLY.value, Path.FLOOR_PLY.value, -pitch_rad, -roll_rad)
        rotate_ply_file_with_colors(Path.DEPTH_PLY.value, Path.DEPTH_PLY.value, -pitch_rad, -roll_rad)

        # Run screened_poisson_surface_reconstruction()
        if Config.BLENDER_ROOM_TYPE.value == "mesh":
            run_subprocess(Path.MESHLAB_SCRIPT.value, Path.DEPTH_PLY.value)

        camera_height = self.estimate_camera_height((pitch_rad, roll_rad))

        scene_render_parameters = dict()
        from math import radians
        scene_render_parameters["camera_location"] = [0, 0, 0]
        scene_render_parameters["camera_angles"] = float(radians(90) - pitch_rad), float(-roll_rad), 0
        scene_render_parameters['resolution_x'] = width
        scene_render_parameters['resolution_y'] = height
        scene_render_parameters['room_point_cloud_path'] = Path.DEPTH_PLY.value
        scene_render_parameters['objects'] = dict()
        scene_render_parameters['focal_length_px'] = self.focal_length_px
        scene_render_parameters['lights'] = self.calculate_light_offsets_and_angles(
            (pitch_rad, roll_rad)
        )

        self.create_floor_layout(pitch_rad, roll_rad)

        return camera_height, pitch_rad, roll_rad, height, scene_render_parameters

    def calculate_curtains_parameters(self, camera_height, camera_angles_rad: tuple):
        from stage.furniture.Curtain import Curtain
        pitch_rad, roll_rad = camera_angles_rad
        curtain = Curtain(2.2, Path.CURTAIN_MODEL.value)
        Room.save_windows_mask(Path.SEG_INPUT_IMAGE.value, Path.WINDOWS_MASK_IMAGE.value)
        pixels_for_placing = curtain.find_placement_pixel(Path.WINDOWS_MASK_IMAGE.value)
        print(f"CURTAINS placement pixels: {pixels_for_placing}")
        curtains_parameters = []
        for window in pixels_for_placing:
            try:
                left_top_point, right_top_point = window
                left_curtain_offset, right_curtain_offset = [self.infer_3d(pixel, pitch_rad, roll_rad) for
                                                             pixel in (left_top_point, right_top_point)]
                yaw_angle = calculate_angle_from_top_view(left_curtain_offset, right_curtain_offset)
                for pixel in (left_top_point, right_top_point):
                    render_parameters = curtain.calculate_rendering_parameters(self, pixel, yaw_angle,
                                                                               (roll_rad, pitch_rad))
                    # WARNING!
                    # We set both left and right curtains height equal to the height of the left curtain.
                    # So we avoid differences in their height level attachment
                    # If you want to avoid it and calculate attachment for each separately:
                    # curtains_height = camera_height + render_parameters['obj_offsets'][2]
                    render_parameters['obj_offsets'] = (render_parameters['obj_offsets'][0],
                                                        render_parameters['obj_offsets'][1],
                                                        left_curtain_offset[2])
                    curtains_height = camera_height + left_curtain_offset[2]

                    height_scale = curtain.calculate_height_scale(curtains_height)
                    render_parameters['obj_scale'] = (render_parameters['obj_scale'][0],
                                                      render_parameters['obj_scale'][1],
                                                      height_scale)
                    curtains_parameters.append(render_parameters)

            except IndexError as e:
                print(f"{e}, we skip adding curtains for a window.")
        return curtains_parameters

    def calculate_plant_parameters(self, camera_angles_rad: tuple):
        from stage.furniture.Plant import Plant
        from stage.Floor import Floor
        pitch_rad, roll_rad = camera_angles_rad
        plant = Plant()
        seg_image_path = Path.SEG_INPUT_IMAGE.value
        save_path = Path.FLOOR_MASK_IMAGE.value
        Floor.save_mask(seg_image_path, save_path)
        pixels_for_placing = plant.find_placement_pixel(save_path)
        print(f"PLANT placement pixels: {pixels_for_placing}")
        import random
        random_index = random.randint(0, len(pixels_for_placing) - 1)
        plant_yaw_angle = 0  # We do not rotate plants
        render_parameters = (
            plant.calculate_rendering_parameters(self, pixels_for_placing[random_index], plant_yaw_angle,
                                                 (roll_rad, pitch_rad)))
        return render_parameters

    @staticmethod
    def find_window_boundaries(mask_image_path):
        img = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
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

            # Finding the extreme points for each contour
            top_left = contour_points[np.argmin(contour_points[:, 0] + contour_points[:, 1])]
            top_right = contour_points[np.argmax(contour_points[:, 0] - contour_points[:, 1])]
            bottom_point = contour_points[np.argmax(contour_points[:, 1])]  # Lowest point in y-axis

            M = cv2.moments(contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid = (cx, cy)

            print("Corner coordinates:")
            print("Top-left:", top_left)
            print("Top-right:", top_right)
            print("Bottom:", bottom_point)

            # Append the triplet of points to the final list
            final_points.append((top_left, top_right, bottom_point, centroid))
        # Return the list of points for all windows
        return final_points

    def calculate_light_offsets_and_angles(self, camera_angles_rad: tuple):
        """
        Calculates and returns light parameters based on instance attributes.
        """
        pitch_rad, roll_rad = camera_angles_rad

        # Retrieve window boundaries instead of centroids
        window_boundaries = Room.find_window_boundaries(Path.WINDOWS_MASK_IMAGE.value)

        light_parameters = []

        for boundary_points in window_boundaries:
            try:
                # Unpack the boundary points returned by find_window_boundaries
                top_left, top_right, bottom_point, centroid = boundary_points

                # Assign left and right top points for light placement calculations
                left_top_point = top_left
                right_top_point = top_right

                # Convert boundary points to 3D space with infer_3d
                left_light_offset, right_light_offset, bottom_offset, window_centroid_offset = [
                    self.infer_3d(pixel, pitch_rad, roll_rad) for pixel in
                    (left_top_point, right_top_point, bottom_point, centroid)
                ]

                # Move light a bit behind the window
                window_centroid_offset[1] -= 0.15
                window_centroid_offset[0] -= 0.15
                # Calculate the yaw angle, window_width, window_height for light orientation and size
                yaw_angle = calculate_angle_from_top_view(left_light_offset, right_light_offset)
                window_width = np.linalg.norm(np.array(right_light_offset) - np.array(left_light_offset))
                window_height = np.linalg.norm(np.array(bottom_offset) - np.array(left_light_offset))
                print(f' Window Width: {window_width}'
                      f'Window Height: {window_height}'
                      f'Offsets: Right Light = {right_light_offset}, Left Light = {left_light_offset}, '
                      f'Bottom = {bottom_offset}, Centroid = {window_centroid_offset}')

                # Append light parameters for the current window
                light_parameters.append({
                    'offset': window_centroid_offset,
                    'yaw_angle': int(yaw_angle),
                    'size': window_width,
                    'size_y': window_height
                })
            except IndexError as e:
                print(f"{e}, skipping this window.")
        return light_parameters
