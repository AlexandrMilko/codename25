from preprocessing.preProcessNormalMap import ImageNormalMap
from preprocessing.preProcessSegment import ImageSegmentor
from stage import Floor
from tools import (move_file, copy_file, get_image_size, save_mask_of_size,
                   convert_png_to_mask, overlay_masks, image_overlay, calculate_angle_from_top_view,
                   resize_and_save_image)
from constants import Path
from PIL import Image
import open3d as o3d
import numpy as np
import cv2
import os


class Room:
    # BGR, used in segmented images
    window_color = (230, 230, 230)
    door_color = (51, 255, 8)
    floor_color = (50, 50, 80)
    blind_color = (255, 61, 0)  # blind that is set on windows, kinda curtains

    def __init__(self, empty_room_image_path):  # Original image path is an empty space image
        self.empty_room_image_path = empty_room_image_path

    def find_roll_pitch(self) -> tuple[float, float]:
        width, height = get_image_size(self.empty_room_image_path)
        PREPROCESSOR_RESOLUTION_LIMIT = 1024 if height > 1024 else height

        normalMap = ImageNormalMap(self.empty_room_image_path, Path.PREPROCESSED_USERS.value, PREPROCESSOR_RESOLUTION_LIMIT)
        normalMap.execute()

        copy_file(self.empty_room_image_path,
                  "UprightNet/imgs/rgb/users.png")  # We copy it because we will use it later in get_wall method and we want to have access to the image
        move_file(f"images/preprocessed/users.png",
                  "UprightNet/imgs/normal_pair/users.png")
        from UprightNet.infer import get_roll_pitch
        try:
            return get_roll_pitch()
        except Exception as e:
            os.chdir('..')
            print(f"EXCEPTION: {e}")
            print("Returning default angles")
            return 0, 0

    @staticmethod
    def get_walls():
        import stage.Wall
        return stage.Wall.find_walls(Path.SEGMENTED_ES_IMAGE.value)

    @staticmethod
    def get_biggest_wall():
        import stage.Wall
        return stage.Wall.find_biggest_wall(Path.SEGMENTED_ES_IMAGE.value)

    def infer_3d(self, pixel: tuple[int, int], pitch_rad: float, roll_rad: float):
        from DepthAnythingV2.depth_estimation import image_pixel_to_3d, rotate_3d_point
        print(self.empty_room_image_path, pixel, "IMAGE PATH and PIXEL")
        target_point = image_pixel_to_3d(*pixel, self.empty_room_image_path)
        # We rotate it back to compensate our camera rotation
        offset_relative_to_camera = rotate_3d_point(target_point, -pitch_rad, -roll_rad)
        return offset_relative_to_camera

    def pixel_mapping_floor_layout(self, pitch_rad: float, roll_rad: float):
        horizontal_borders = self.find_horizontal_borders()
        print(horizontal_borders)

        points_in_3d = {}
        for name, value in horizontal_borders.items():
            points_in_3d[name] = []
            left_point = self.infer_3d(value[0], pitch_rad, roll_rad)
            right_point = self.infer_3d(value[1], pitch_rad, roll_rad)
            points_in_3d[name].append(left_point)
            points_in_3d[name].append(right_point)

        print(points_in_3d)
        Room.offsets_to_floor_pixels(Path.FLOOR_PLY.value, points_in_3d)
        
    @staticmethod
    def pixel_to_3d(x, y):
        """
        Args:
            x: x coordinate of the pixel
            y: y coordinate of the pixel
            # filename: name of file

        Returns:
            X_3D, Y_3D: coordinates of pixel
        """
        # Load the layout image to get dimensions
        layout_image = Image.open(Path.FLOOR_LAYOUT_IMAGE.value).convert('RGB')
        original_width, original_height = layout_image.size

        # Load the depth map
        depth_image = np.load(Path.DEPTH_IMAGE.value)

        # Ensure we are getting the correct depth value
        resized_depth = Image.fromarray(depth_image).resize((original_width, original_height), Image.NEAREST)
        Z_depth = np.array(resized_depth)[y, x]

        # Compute focal lengths based on original image dimensions
        FX = original_width * 0.6

        # Compute 3D coordinates
        X_3D = (x - original_width / 2.1) * Z_depth / FX
        Z_3D = Z_depth / 1.3

        return [X_3D, Z_3D, 0]

    def estimate_camera_height(self, camera_angles: tuple[float, float]):
        pitch, roll = camera_angles
        from DepthAnythingV2.depth_estimation import rotate_3d_point, image_pixel_to_3d
        import stage.Floor
        floor_pixel = stage.Floor.find_centroid(Path.SEGMENTED_ES_IMAGE.value)
        point_3d = image_pixel_to_3d(*floor_pixel, self.empty_room_image_path)
        print(f"Floor Centroid: {floor_pixel} -> {point_3d}")
        rotated_point = rotate_3d_point(point_3d, -pitch, -roll)
        z_coordinate = rotated_point[2]
        return abs(z_coordinate)

    @staticmethod
    def find_number_of_windows(windows_mask_path: str) -> int:
        # Загрузка изображения
        img = cv2.imread(windows_mask_path, cv2.IMREAD_GRAYSCALE)  # Укажите правильный путь к файлу
        blurred = cv2.GaussianBlur(img, (5, 5), 0)

        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(thresh, kernel, iterations=1)
        img = cv2.dilate(erosion, kernel, iterations=1)

        # Поиск контуров
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

        # Define the lower and upper bounds for the color
        tolerance = 3
        window_lower_color = np.array([x - tolerance for x in window_rgb_values])
        window_upper_color = np.array([x + tolerance for x in window_rgb_values])
        blind_lower_color = np.array([x - tolerance for x in blind_rgb_values])
        blind_upper_color = np.array([x + tolerance for x in blind_rgb_values])

        # Create a mask for the color
        window_color_mask = cv2.inRange(image, window_lower_color, window_upper_color)
        blind_color_mask = cv2.inRange(image, blind_lower_color, blind_upper_color)

        combined_mask = cv2.bitwise_or(window_color_mask, blind_color_mask)

        _, thresh = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)
        width, height = get_image_size(segmented_image_path)
        kernel = np.ones((height // 25, height // 25), np.uint8)  # We adjust kernel based on img size
        print(width, height, "WINDOWS IMG WIDTH, HEIGHT")
        print(height // 25, "KERNEL SIZE for window mask denoising")
        erosion = cv2.erode(thresh, kernel, iterations=1)
        color_mask = cv2.dilate(erosion, kernel, iterations=1)
        # cv2.imshow('gray', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Create a black and white mask
        bw_mask = np.zeros_like(color_mask)
        bw_mask[color_mask != 0] = 255

        # Check if the mask contains white pixels
        cv2.imwrite(output_windows_mask_path, bw_mask)

    @staticmethod
    def find_horizontal_borders():
        target_colors = {
            "door": np.array([8, 255, 51]),  # #08FF33
            "window": np.array([230, 230, 230]),  # #E6E6E6
            # Add more colors here as needed
        }

        image = cv2.imread(Path.SEGMENTED_ES_IMAGE.value)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bottom_pixels = {}
        label_counters = {label: 0 for label in target_colors.keys()}  # Initialize counters for each label

        min_area_threshold = 800  # Adjust this value as needed

        for object_name, color_value in target_colors.items():
            # Create mask for the current color
            mask = cv2.inRange(image_rgb, color_value, color_value)
            cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

            # Find contours in the cleaned mask
            contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > min_area_threshold:
                    leftmost_bottom_pixel = tuple(contour[contour[:, :, 0].argmin()][0])
                    rightmost_bottom_pixel = tuple(contour[contour[:, :, 0].argmax()][0])

                    # Increment the counter for the current label
                    label_counters[object_name] += 1
                    # Create a unique label for the current object
                    unique_label = f"{object_name}{label_counters[object_name]}"
                    # Store the pixel coordinates in the bottom_pixels dictionary
                    bottom_pixels[unique_label] = (leftmost_bottom_pixel, rightmost_bottom_pixel)

        return bottom_pixels

    @staticmethod
    def offsets_to_floor_pixels(ply_path, points_dict, output_path=Path.FLOOR_LAYOUT_IMAGE.value) -> (
    dict, tuple):
        """
        :param ply_path: path to the .ply file that represents 3d space
        :param output_path: path to save the debug layout image
        :param points_dict: 3d points to be converted into 2d layout pixel coords
        :return: dictionary of converted 2D points and pixels-per-meter ratio
        """

        # Load the point cloud
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)

        # WARNING: Ensure path only contains floor points
        floor_points = points

        # Initialize layout image
        height, width = 1024, 1024
        layout_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Add camera and relative point for pixel-per-meter calculation
        points_dict['camera'] = [[0, 0, 0], [0, 0, 0]]
        points_dict['point_for_calculating_ratio'] = [[0.2, 0.2, 0], [0.2, 0.2, 0]]

        # Process user's points
        for point_name in points_dict.keys():
            print(points_dict[point_name], " points_dict[point_name]")
            left = points_dict[point_name][0]
            right = points_dict[point_name][1]

            left[1], left[2] = left[2], left[1]
            right[1], right[2] = right[2], right[1]

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
        norm_points[:, 2] = norm_points[:, 2] * (height - 1)

        hull = cv2.convexHull(norm_points[:, [0, 2]].astype(int))
        cv2.fillPoly(layout_image, [hull], (255, 255, 255))

        # Print normalized points for debugging
        print("Normalized points (first 5):", norm_points[:5])

        # # Visualize all points on the image
        # for point in norm_points:
        #     pixel_x = int(point[0])
        #     pixel_y = int(point[2])
        #     cv2.circle(layout_image, (pixel_x, pixel_y), 1, (255, 255, 255), -1)  # White color for all points

        # Convert 3D points to 2D pixels
        result = dict()
        for point_name in points_dict.keys():
            print(points_dict)
            left = points_dict[point_name][0]
            right = points_dict[point_name][1]
            result[point_name] = []
            for point in left, right:
                x_3d, _, y_3d = point
                print(f"3D Point: {point}")
                pixel_x = int((x_3d - min_coords[0]) / (max_coords[0] - min_coords[0]) * (width - 1))
                pixel_y = int((y_3d - min_coords[2]) / (max_coords[2] - min_coords[2]) * (height - 1))

                # Ensure pixel coordinates are within bounds
                pixel_x = np.clip(pixel_x, 0, width - 1)
                pixel_y = np.clip(pixel_y, 0, height - 1)
                print(f"Mapped to 2D: x={pixel_x}, y={pixel_y}")  # Debug message
                result[point_name].append([pixel_x, pixel_y])
                cv2.circle(layout_image, (pixel_x, pixel_y), 5, (0, 0, 255), -1)  # Red color for specific points

        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, layout_image)

        pixels_per_meter_ratio = Room.calculate_pixels_per_meter_ratio(points_dict, result)
        print(pixels_per_meter_ratio)

        return result, pixels_per_meter_ratio

    @staticmethod
    def calculate_pixels_per_meter_ratio(offsets, pixels):
        """
        offsets: points in 3d space that were converted to the pixels in dictionary format
        pixels: pixel coordinates on floor layout image as result of conversion in dictionary format
        WARNING! Both dictionaries must have 'camera' and 'point_for_calculating_ratio' keys
        """
        left_camera_pixel = pixels['camera'][0]
        left_point_pixel = pixels['point_for_calculating_ratio'][0]

        left_camera_offset = offsets['camera'][0]
        left_point_offset = offsets['point_for_calculating_ratio'][0]
        pixels_x_diff = left_camera_pixel[0] - left_point_pixel[0]
        pixels_y_diff = left_camera_pixel[1] - left_point_pixel[1]
        offsets_x_diff = left_camera_offset[0] - left_point_offset[0]
        offsets_z_diff = left_camera_offset[2] - left_point_offset[2]
        ratio_x = pixels_x_diff / offsets_x_diff
        ratio_y = pixels_y_diff / offsets_z_diff
        return ratio_x, ratio_y

    @staticmethod
    def calculate_offset_from_pixel_diff(pixels_diff, ratio):
        """
        pixels_diff: has the following format [pixels_x_diff, pixels_y_diff].
        Represents difference in pixel coordinates between 2 points

        ratio: has the following format [ratio_x, ratio_y]
        Represents pixels_per_meter_ratio for a floor layout. For both axes.
        """
        pixel_x_diff, pixel_y_diff = pixels_diff
        ratio_x, ratio_y = ratio
        offset_x, offset_z = pixel_x_diff / ratio_x, pixel_y_diff / ratio_y
        return offset_x, offset_z

    def calculate_curtains_parameters(self, camera_height, camera_angles_rad: tuple):
        from stage.furniture.Curtain import Curtain
        pitch_rad, roll_rad = camera_angles_rad
        curtain = Curtain(2.2, Path.CURTAIN_MODEL.value)
        Room.save_windows_mask(Path.SEGMENTED_ES_IMAGE.value, Path.WINDOWS_MASK_IMAGE.value)
        pixels_for_placing = curtain.find_placement_pixel(Path.WINDOWS_MASK_IMAGE.value)
        print(f"CURTAINS placement pixels: {pixels_for_placing}")
        curtains_parameters = []
        for window in pixels_for_placing:
            try:
                left_top_point, right_top_point = window
                yaw_angle = calculate_angle_from_top_view(*[self.infer_3d(pixel, pitch_rad, roll_rad) for
                                                            pixel in (left_top_point, right_top_point)])
                for pixel in (left_top_point, right_top_point):
                    render_parameters = curtain.calculate_rendering_parameters(self, pixel, yaw_angle,
                                                                               (roll_rad, pitch_rad))
                    curtains_height = camera_height + render_parameters['obj_offsets'][2]
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
        seg_image_path = Path.SEGMENTED_ES_IMAGE.value
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

    def prepare_empty_room_data(self):
        Image.open(self.empty_room_image_path).save(Path.PREREQUISITE_IMAGE.value)
        from DepthAnythingV2.depth_estimation import (image_pixels_to_point_cloud, depth_ply_path, floor_ply_path,
                                                      create_floor_point_cloud, rotate_ply_file_with_colors)
        roll_rad, pitch_rad = np.negative(self.find_roll_pitch())

        image_pixels_to_point_cloud(self.empty_room_image_path)
        # image_pixels_to_3d(self.empty_room_image_path, "my_3d_space.txt")
        # rotate_3d_points("my_3d_space.txt", "my_3d_space_rotated.txt", -pitch_rad, -roll_rad)

        # Segment our empty space room. It is used in Room.save_windows_mask
        width, height = get_image_size(self.empty_room_image_path)
        PREPROCESSOR_RESOLUTION_LIMIT = 1024 if height > 1024 else height

        segment = ImageSegmentor(self.empty_room_image_path, Path.SEGMENTED_ES_IMAGE.value, PREPROCESSOR_RESOLUTION_LIMIT)
        segment.execute()
        resize_and_save_image(Path.SEGMENTED_ES_IMAGE.value,
                              Path.SEGMENTED_ES_IMAGE.value, height)

        Floor.save_mask(Path.SEGMENTED_ES_IMAGE.value, Path.FLOOR_MASK_IMAGE.value)
        create_floor_point_cloud(self.empty_room_image_path)
        rotate_ply_file_with_colors(floor_ply_path, floor_ply_path, -pitch_rad, -roll_rad)
        camera_height = self.estimate_camera_height([pitch_rad, roll_rad])

        # Create an empty mask of same size as image
        mask_path = Path.FURNITURE_MASK_IMAGE.value
        width, height = get_image_size(self.empty_room_image_path)
        save_mask_of_size(width, height, mask_path)

        scene_render_parameters = dict()
        from math import radians
        scene_render_parameters["camera_location"] = [0, 0, 0]
        # We set opposite
        # We add 90 to the pitch, because originally camera is rotated pointing downwards in Blender
        scene_render_parameters["camera_angles"] = float(radians(90) - pitch_rad), float(
            +roll_rad), 0  # We convert to float to avoid JSON conversion errors from numpy
        scene_render_parameters['resolution_x'] = width
        scene_render_parameters['resolution_y'] = height
        scene_render_parameters['objects'] = dict()

        return camera_height, pitch_rad, roll_rad, height, scene_render_parameters

    @staticmethod
    def process_rendered_image(furniture_image):
        furniture_image.save(Path.FURNITURE_PIECE_MASK_IMAGE.value)
        convert_png_to_mask(Path.FURNITURE_PIECE_MASK_IMAGE.value)
        overlay_masks(Path.FURNITURE_PIECE_MASK_IMAGE.value, Path.FURNITURE_MASK_IMAGE.value,
                      Path.FURNITURE_MASK_IMAGE.value)
        background_image = Image.open(Path.PREREQUISITE_IMAGE.value)
        combined_image = image_overlay(furniture_image, background_image)
        combined_image.save(Path.PREREQUISITE_IMAGE.value)
