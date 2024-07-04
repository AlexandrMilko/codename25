import cv2
from PIL import Image
import os
from tools import move_file, run_preprocessor, copy_file, convert_to_mask, overlay_images, create_furniture_mask, get_image_size
import numpy as np
import open3d as o3d

class Room:
    # BGR, used in segmented images
    window_color = (230, 230, 230)
    door_color = (51, 255, 8)
    floor_color = (50, 50, 80)
    blind_color = (255, 61, 0)  # blind that is set on windows, kinda curtains
    def __init__(self, empty_room_image_path): # Original image path is an empty space image
        self.empty_room_image_path = empty_room_image_path

    def find_roll_pitch(self) -> tuple[float, float]:
        width, height = get_image_size(self.empty_room_image_path)
        run_preprocessor("normal_dsine", self.empty_room_image_path, "users.png", height)
        copy_file(self.empty_room_image_path, "UprightNet/imgs/rgb/users.png") # We copy it because we will use it later in get_wall method and we want to have access to the image
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
        return stage.Wall.find_walls(f'images/preprocessed/segmented_es.png')

    def get_biggest_wall(self):
        width, height = get_image_size(self.empty_room_image_path)
        run_preprocessor("seg_ofade20k", self.empty_room_image_path, "segmented_es.png", height)
        import stage.Wall
        return stage.Wall.find_biggest_wall(f'images/preprocessed/segmented_es.png')

    def infer_3d(self, pixel: tuple[int, int], pitch_rad: float, roll_rad: float):
        from DepthAnything.depth_estimation import image_pixel_to_3d, rotate_3d_point
        print(self.empty_room_image_path, pixel, "IMAGE PATH and PIXEL")
        target_point = image_pixel_to_3d(*pixel, self.empty_room_image_path)
        # We rotate it back to compensate our camera rotation
        offset_relative_to_camera = rotate_3d_point(target_point, -pitch_rad, -roll_rad)
        return offset_relative_to_camera

    def estimate_camera_height(self, camera_angles: tuple[float, float]):
        pitch, roll = camera_angles
        from DepthAnything.depth_estimation import rotate_3d_point, image_pixel_to_3d
        import stage.Floor
        floor_pixel = stage.Floor.find_centroid(f'images/preprocessed/segmented_es.png')
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
    def save_windows_mask(segmented_image_path: str, windows_mask_path: str):
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
        _, height = get_image_size(windows_mask_path)
        kernel = np.ones((height // 25, height // 25), np.uint8) # We adjust kernel based on img size
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
        cv2.imwrite(windows_mask_path, bw_mask)

    @staticmethod
    def save_floor_layout_image(ply_path: str, npy_path: str, output_path: str) -> None:
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
        from tools import calculate_angle_from_top_view, get_image_size, convert_png_to_mask, overlay_masks, image_overlay
        pitch_rad, roll_rad = camera_angles_rad
        curtain = Curtain()
        segmented_es_path = f'images/preprocessed/segmented_es.png'
        Room.save_windows_mask(segmented_es_path, f'images/preprocessed/windows_mask.png')
        pixels_for_placing = curtain.find_placement_pixel(
            f'images/preprocessed/windows_mask.png')
        print(f"CURTAINS placement pixels: {pixels_for_placing}")
        Image.open(self.empty_room_image_path).save(prerequisite_path)
        for window in pixels_for_placing:
            try:
                left_top_point, right_top_point = window
                yaw_angle = calculate_angle_from_top_view(*[self.infer_3d(pixel, pitch_rad, roll_rad) for
                                                            pixel in (left_top_point, right_top_point)])
                for pixel in (left_top_point, right_top_point):
                    render_parameters = curtain.calculate_rendering_parameters(self, pixel, yaw_angle,
                                                                               (roll_rad, pitch_rad))
                    width, height = get_image_size(self.empty_room_image_path)
                    render_parameters['resolution_x'] = width
                    render_parameters['resolution_y'] = height
                    curtains_height = camera_height + render_parameters['obj_offsets'][2]
                    curtains_height_scale = curtains_height / Curtain.default_height
                    render_parameters['obj_scale'] = render_parameters['obj_scale'][0], render_parameters['obj_scale'][
                        1], curtains_height_scale
                    curtain_image = curtain.request_blender_render(render_parameters)
                    curtain_image.save(tmp_mask_path)
                    convert_png_to_mask(tmp_mask_path)
                    overlay_masks(tmp_mask_path, mask_path, mask_path, [0, 0])
                    background_image = Image.open(prerequisite_path)
                    combined_image = image_overlay(curtain_image, background_image)
                    combined_image.save(prerequisite_path)
            except IndexError as e:
                print(f"{e}, we skip adding curtains for a window.")

    def add_plant(self, camera_angles_rad: tuple, mask_path, tmp_mask_path, prerequisite_path):
        from stage.furniture.Plant import Plant
        from stage.Floor import Floor
        from tools import convert_png_to_mask, image_overlay, overlay_masks
        pitch_rad, roll_rad = camera_angles_rad
        plant = Plant()
        seg_image_path = f'images/preprocessed/segmented_es.png'
        save_path = 'images/preprocessed/floor_mask.png'
        Floor.save_mask(seg_image_path, save_path)
        pixels_for_placing = plant.find_placement_pixel(save_path)
        print(f"PLANT placement pixels: {pixels_for_placing}")
        import random
        random_index = random.randint(0, len(pixels_for_placing) - 1)
        plant_yaw_angle = 0  # We do not rotate plants
        render_parameters = (
            plant.calculate_rendering_parameters(self, pixels_for_placing[random_index], plant_yaw_angle,
                                                 (roll_rad, pitch_rad)))
        width, height = get_image_size(self.empty_room_image_path)
        render_parameters['resolution_x'] = width
        render_parameters['resolution_y'] = height
        plant_image = plant.request_blender_render(render_parameters)
        plant_image.save(tmp_mask_path)
        convert_png_to_mask(tmp_mask_path)
        overlay_masks(tmp_mask_path, mask_path, mask_path, [0, 0])
        background_image = Image.open(prerequisite_path)
        combined_image = image_overlay(plant_image, background_image)
        combined_image.save(prerequisite_path)

    def add_bed(self, camera_angles_rad: tuple, mask_path, tmp_mask_path, prerequisite_path):
        from stage.furniture.Bed import Bed
        from tools import convert_png_to_mask, image_overlay, overlay_masks
        pitch_rad, roll_rad = camera_angles_rad
        bed = Bed()
        wall = self.get_biggest_wall()
        render_directory = f'images/preprocessed'
        wall.save_mask(os.path.join(render_directory, 'wall_mask.png'))
        pixel_for_placing = bed.find_placement_pixel(os.path.join(render_directory, 'wall_mask.png'))
        print(f"BED placement pixel: {pixel_for_placing}")
        yaw_angle = wall.find_angle_from_3d(self, pitch_rad, roll_rad)
        render_parameters = (
            bed.calculate_rendering_parameters(self, pixel_for_placing, yaw_angle, (roll_rad, pitch_rad)))
        width, height = get_image_size(self.empty_room_image_path)
        render_parameters['resolution_x'] = width
        render_parameters['resolution_y'] = height
        bed_image = bed.request_blender_render(render_parameters)
        bed_image.save(tmp_mask_path)
        convert_png_to_mask(tmp_mask_path)
        overlay_masks(tmp_mask_path, mask_path, mask_path, [0, 0])
        background_image = Image.open(prerequisite_path)
        combined_image = image_overlay(bed_image, background_image)
        combined_image.save(prerequisite_path)

    def add_sofa_with_table(self, camera_angles_rad: tuple, mask_path, tmp_mask_path, prerequisite_path):
        from stage.furniture.SofaWithTable import SofaWithTable
        from tools import convert_png_to_mask, image_overlay, overlay_masks
        pitch_rad, roll_rad = camera_angles_rad
        sofa_with_table = SofaWithTable()
        wall = self.get_biggest_wall()
        render_directory = f'images/preprocessed/'
        wall.save_mask(os.path.join(render_directory, 'wall_mask.png'))
        pixel_for_placing = sofa_with_table.find_placement_pixel(os.path.join(render_directory, 'wall_mask.png'))
        print(f"SofaWithTable placement pixel: {pixel_for_placing}")
        yaw_angle = wall.find_angle_from_3d(self, pitch_rad, roll_rad)
        render_parameters = (
            sofa_with_table.calculate_rendering_parameters(self, pixel_for_placing, yaw_angle, (roll_rad, pitch_rad)))
        width, height = get_image_size(self.empty_room_image_path)
        render_parameters['resolution_x'] = width
        render_parameters['resolution_y'] = height
        sofa_image = sofa_with_table.request_blender_render(render_parameters)
        sofa_image.save(tmp_mask_path)
        convert_png_to_mask(tmp_mask_path)
        overlay_masks(tmp_mask_path, mask_path, mask_path, [0, 0])
        background_image = Image.open(prerequisite_path)
        combined_image = image_overlay(sofa_image, background_image)
        combined_image.save(prerequisite_path)

    def add_kitchen_table_with_chairs(self, camera_angles_rad: tuple, mask_path, tmp_mask_path, prerequisite_path):
        from stage.furniture.KitchenTableWithChairs import KitchenTableWithChairs
        from stage.Floor import Floor
        from tools import convert_png_to_mask, image_overlay, overlay_masks
        import random
        pitch_rad, roll_rad = camera_angles_rad

        kitchen_table_with_chairs = KitchenTableWithChairs()
        seg_image_path = f'images/preprocessed/segmented_es.png'
        save_path = 'images/preprocessed/floor_mask.png'
        Floor.save_mask(seg_image_path, save_path)

        kitchen_table_with_chairs.find_placement_pixel_from_floor_layout('images/preprocessed/floor_layout.png')

        pixels_for_placing = kitchen_table_with_chairs.find_placement_pixel(save_path)
        print(f"KitchenTableWithChairs placement pixel: {pixels_for_placing}")
        wall = self.get_biggest_wall()
        render_directory = f'images/preprocessed/'
        wall.save_mask(os.path.join(render_directory, 'wall_mask.png'))
        yaw_angle = wall.find_angle_from_3d(self, pitch_rad, roll_rad)
        random_index = random.randint(0, len(pixels_for_placing) - 1)
        render_parameters = (
            kitchen_table_with_chairs.calculate_rendering_parameters(self, pixels_for_placing[random_index], yaw_angle,
                                                                     (roll_rad, pitch_rad)))
        width, height = get_image_size(self.empty_room_image_path)
        render_parameters['resolution_x'] = width
        render_parameters['resolution_y'] = height
        table_image = kitchen_table_with_chairs.request_blender_render(render_parameters)
        table_image.save(tmp_mask_path)
        convert_png_to_mask(tmp_mask_path)
        overlay_masks(tmp_mask_path, mask_path, mask_path, [0, 0])
        background_image = Image.open(prerequisite_path)
        combined_image = image_overlay(table_image, background_image)
        combined_image.save(prerequisite_path)

        # Create windows mask for staged room
        run_preprocessor("seg_ofade20k", prerequisite_path, "seg_prerequisite.png", height)
        segmented_es_path = f'images/preprocessed/seg_prerequisite.png'
        Room.save_windows_mask(segmented_es_path,
                               f'images/preprocessed/windows_mask_inpainting.png')