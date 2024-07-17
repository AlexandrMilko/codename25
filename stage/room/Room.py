from tools import (move_file, run_preprocessor, copy_file, get_image_size, save_mask_of_size,
                   convert_png_to_mask, overlay_masks, image_overlay, calculate_angle_from_top_view)
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
        run_preprocessor("normal_dsine", self.empty_room_image_path, "users.png", height)
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

    def get_walls(self):
        width, height = get_image_size(self.empty_room_image_path)
        run_preprocessor("seg_ofade20k", self.empty_room_image_path, "segmented_es.png", height)
        import stage.Wall
        return stage.Wall.find_walls(Path.SEGMENTED_ES_IMAGE.value)

    def get_biggest_wall(self):
        width, height = get_image_size(self.empty_room_image_path)
        run_preprocessor("seg_ofade20k", self.empty_room_image_path, "segmented_es.png", height)
        import stage.Wall
        return stage.Wall.find_biggest_wall(Path.SEGMENTED_ES_IMAGE.value)

    def infer_3d(self, pixel: tuple[int, int], pitch_rad: float, roll_rad: float):
        from DepthAnything.depth_estimation import image_pixel_to_3d, rotate_3d_point
        print(self.empty_room_image_path, pixel, "IMAGE PATH and PIXEL")
        target_point = image_pixel_to_3d(*pixel, self.empty_room_image_path)
        # We rotate it back to compensate our camera rotation
        offset_relative_to_camera = rotate_3d_point(target_point, -pitch_rad, -roll_rad)
        return offset_relative_to_camera

    def pixel_to_3d(self, x, y):
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

        return X_3D, Z_3D, 0

    def estimate_camera_height(self, camera_angles: tuple[float, float]):
        pitch, roll = camera_angles
        from DepthAnything.depth_estimation import rotate_3d_point, image_pixel_to_3d
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
        kernel = np.ones((height // 25, height // 25), np.uint8) # We adjust kernel based on img size
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
    def save_floor_layout_image(ply_path: str, npy_path: str, output_path=Path.FLOOR_LAYOUT_IMAGE.value) -> None:
        # Загрузка облака точек
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)

        # Фильтрация точек пола
        quantile = 60
        floor_height = np.percentile(points[:, 1], quantile)
        threshold = 0.05  # Допустимое отклонение от высоты пола
        floor_points = points[np.abs(points[:, 1] - floor_height) < threshold]

        # Загрузка карты глубины
        depth_map = np.load(npy_path)
        height, width = depth_map.shape
        layout_image = np.zeros((height, width), dtype=np.uint8)

        # Нахождение минимальных и максимальных значений координат пола
        min_coords = floor_points.min(axis=0)
        max_coords = floor_points.max(axis=0)

        # Нормализация координат пола
        norm_points = (floor_points - min_coords) / (max_coords - min_coords)
        norm_points[:, 0] = norm_points[:, 0] * (width - 1)
        norm_points[:, 2] = norm_points[:, 2] * (height - 1)

        # Создание заполненного контура
        hull = cv2.convexHull(norm_points[:, [0, 2]].astype(int))
        cv2.fillPoly(layout_image, [hull], (255, 255, 255))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, layout_image)

        # Визуализация результатов
        # cv2.imshow('Layout Image', layout_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def add_curtains(self, camera_height, camera_angles_rad: tuple, mask_path, tmp_mask_path, prerequisite_path):
        from stage.furniture.Curtain import Curtain
        pitch_rad, roll_rad = camera_angles_rad
        curtain = Curtain(2.2, Path.CURTAIN_MODEL.value)
        Room.save_windows_mask(Path.SEGMENTED_ES_IMAGE.value, Path.WINDOWS_MASK_IMAGE.value)
        pixels_for_placing = curtain.find_placement_pixel(Path.WINDOWS_MASK_IMAGE.value)
        print(f"CURTAINS placement pixels: {pixels_for_placing}")
        Image.open(self.empty_room_image_path).save(prerequisite_path)
        curtains_parameters = []
        for window in pixels_for_placing:
            try:
                left_top_point, right_top_point = window
                yaw_angle = calculate_angle_from_top_view(*[self.infer_3d(pixel, pitch_rad, roll_rad) for
                                                            pixel in (left_top_point, right_top_point)])
                for pixel in (left_top_point, right_top_point):
                    render_parameters = curtain.calculate_rendering_parameters(self, pixel, yaw_angle,
                                                                               (roll_rad, pitch_rad))
                    # width, height = get_image_size(self.empty_room_image_path)
                    # render_parameters['resolution_x'] = width
                    # render_parameters['resolution_y'] = height
                    curtains_height = camera_height + render_parameters['obj_offsets'][2]
                    height_scale = curtain.calculate_height_scale(curtains_height)
                    render_parameters['obj_scale'] = (render_parameters['obj_scale'][0],
                                                      render_parameters['obj_scale'][1],
                                                      height_scale)
                    curtains_parameters.append(render_parameters)
                    # curtain_image = curtain.request_blender_render(render_parameters)
                    # curtain_image.save(tmp_mask_path)
                    # convert_png_to_mask(tmp_mask_path)
                    # overlay_masks(tmp_mask_path, mask_path, mask_path)
                    # background_image = Image.open(prerequisite_path)
                    # combined_image = image_overlay(curtain_image, background_image)
                    #
                    # combined_image.save(prerequisite_path)

            except IndexError as e:
                print(f"{e}, we skip adding curtains for a window.")
        return curtains_parameters

    def add_plant(self, camera_angles_rad: tuple, mask_path, tmp_mask_path, prerequisite_path):
        from stage.furniture.Plant import Plant
        from stage.Floor import Floor
        pitch_rad, roll_rad = camera_angles_rad
        plant = Plant()
        seg_image_path = Path.SEGMENTED_ES_IMAGE.value
        save_path = Path.FLOOR_MASK_IMAGE
        Floor.save_mask(seg_image_path, save_path)
        pixels_for_placing = plant.find_placement_pixel(save_path)
        print(f"PLANT placement pixels: {pixels_for_placing}")
        import random
        random_index = random.randint(0, len(pixels_for_placing) - 1)
        plant_yaw_angle = 0  # We do not rotate plants
        render_parameters = (
            plant.calculate_rendering_parameters(self, pixels_for_placing[random_index], plant_yaw_angle,
                                                 (roll_rad, pitch_rad)))
        # width, height = get_image_size(self.empty_room_image_path)
        # render_parameters['resolution_x'] = width
        # render_parameters['resolution_y'] = height
        return render_parameters
        plant_image = plant.request_blender_render(render_parameters)
        plant_image.save(tmp_mask_path)
        convert_png_to_mask(tmp_mask_path)
        overlay_masks(tmp_mask_path, mask_path, mask_path)
        background_image = Image.open(prerequisite_path)
        combined_image = image_overlay(plant_image, background_image)
        combined_image.save(prerequisite_path)

    def prepare_empty_room_data(self):
        from DepthAnything.depth_estimation import (image_pixels_to_point_cloud, depth_ply_path, depth_npy_path,
                                                    image_pixels_to_3d, rotate_3d_points)
        roll_rad, pitch_rad = np.negative(self.find_roll_pitch())

        image_pixels_to_point_cloud(self.empty_room_image_path)
        self.save_floor_layout_image(depth_ply_path, depth_npy_path)
        # image_pixels_to_3d(self.empty_room_image_path, "my_3d_space.txt")
        # rotate_3d_points("my_3d_space.txt", "my_3d_space_rotated.txt", -pitch_rad, -roll_rad)

        # Segment our empty space room. It is used in Room.save_windows_mask
        width, height = get_image_size(self.empty_room_image_path)
        run_preprocessor("seg_ofade20k", self.empty_room_image_path, "segmented_es.png", height)

        camera_height = self.estimate_camera_height([pitch_rad, roll_rad])

        # Create an empty mask of same size as image
        mask_path = Path.FURNITURE_MASK_IMAGE.value
        width, height = get_image_size(self.empty_room_image_path)
        save_mask_of_size(width, height, mask_path)

        scene_render_parameters = dict()
        from math import radians
        scene_render_parameters["camera_location"] = [0, 0, 0]
        scene_render_parameters["camera_angles"] = float(radians(90) - pitch_rad), float(+roll_rad), 0
        scene_render_parameters['resolution_x'] = width
        scene_render_parameters['resolution_y'] = height
        scene_render_parameters['objects'] = dict()

        return camera_height, pitch_rad, roll_rad, height, scene_render_parameters