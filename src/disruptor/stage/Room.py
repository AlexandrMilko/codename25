import cv2
from PIL import Image
import os
from disruptor.tools import move_file, run_preprocessor, copy_file, convert_to_mask, overlay_images, create_furniture_mask
from disruptor.stage.FurniturePiece import FurniturePiece
from disruptor.stage.Wall import Wall
import numpy as np

class Room:
    # BGR, used in segmented images
    window_color = (230, 230, 230)
    door_color = (51, 255, 8)
    floor_color = (50, 50, 80)
    blind_color = (255, 61, 0)  # blind that is set on windows, kinda curtains
    def __init__(self, original_image_path): # Original image path is an empty space image
        self.original_image_path = original_image_path

    def find_roll_pitch(self) -> tuple[float, float]:
        es_img = Image.open(self.original_image_path)
        width, height = es_img.size
        es_img.close()
        run_preprocessor("normal_dsine", self.original_image_path, "users.png", height)
        copy_file(self.original_image_path, "disruptor/UprightNet/imgs/rgb/users.png") # We copy it because we will use it later in get_wall method and we want to have access to the image
        move_file(f"disruptor/images/preprocessed/users.png",
                  "disruptor/UprightNet/imgs/normal_pair/users.png")
        from disruptor.UprightNet.infer import get_roll_pitch
        return get_roll_pitch()

    def get_walls(self):
        es_img = Image.open(self.original_image_path)
        width, height = es_img.size
        es_img.close()
        run_preprocessor("seg_ofade20k", self.original_image_path, "segmented_es.png", height)
        from disruptor.stage.Wall import Wall
        return Wall.find_walls(f'disruptor/images/preprocessed/segmented_es.png')

    def get_biggest_wall(self):
        es_img = Image.open(self.original_image_path)
        width, height = es_img.size
        es_img.close()
        run_preprocessor("seg_ofade20k", self.original_image_path, "segmented_es.png", height)
        from disruptor.stage.Wall import Wall
        return Wall.find_biggest_wall(f'disruptor/images/preprocessed/segmented_es.png')

    def infer_3d(self, pixel: tuple[int, int], pitch_rad: float, roll_rad: float):
        from disruptor.stage.DepthAnything.depth_estimation import image_pixel_to_3d, rotate_3d_point
        print(self.original_image_path, pixel, "IMAGE PATH and PIXEL")
        target_point = image_pixel_to_3d(*pixel, self.original_image_path)
        # We rotate it back to compensate our camera rotation
        offset_relative_to_camera = rotate_3d_point(target_point, -pitch_rad, -roll_rad)
        return offset_relative_to_camera

    def estimate_camera_height(self, camera_angles: tuple[float, float]):
        pitch, roll = camera_angles
        from disruptor.stage.DepthAnything.depth_estimation import rotate_3d_point, image_pixel_to_3d
        from disruptor.stage.Floor import Floor
        floor_pixel = Floor.find_centroid(f'disruptor/images/preprocessed/segmented_es.png')
        point_3d = image_pixel_to_3d(*floor_pixel, self.original_image_path)
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
        kernel = np.ones((29, 29), np.uint8)
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