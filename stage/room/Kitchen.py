import numpy as np
from constants import Path, Config
from postprocessing.postProcessing import PostProcessor
from preprocessing.preProcessSegment import ImageSegmentor
from run import SD_DOMAIN
from tools import resize_and_save_image, run_preprocessor
from .Room import Room
from ..furniture.Furniture import Furniture
import random
import json


class Kitchen(Room):
    def stage(self):
        camera_height, pitch_rad, roll_rad, height, scene_render_parameters = self.prepare_empty_room_data()

        all_sides = self.floor_layout.find_all_sides_sorted_by_length()

        kitchen_parameters = self.calculate_kitchen_parameters(all_sides, (pitch_rad, roll_rad))
        table_parameters = self.calculate_table_parameters((pitch_rad, roll_rad))
        plant_parameters = self.calculate_plant_parameters((pitch_rad, roll_rad))

        scene_render_parameters['objects'] = [
            kitchen_parameters,
            table_parameters,
            plant_parameters,
        ]
        scene_render_parameters['objects'] = [item for item in scene_render_parameters['objects'] if item is not None]

        print(json.dumps(scene_render_parameters, indent=4))

        Furniture.start_blender_render(scene_render_parameters)

        PREPROCESSOR_RESOLUTION_LIMIT = Config.CONTROLNET_HEIGHT_LIMIT.value if height > Config.CONTROLNET_HEIGHT_LIMIT.value else height

        if Config.UI.value == "comfyui":
            segment = ImageSegmentor(Path.PREREQUISITE_IMAGE.value, Path.SEG_PREREQUISITE_IMAGE.value,
                                     PREPROCESSOR_RESOLUTION_LIMIT)
            segment.execute()
        else:
            run_preprocessor("seg_ofade20k", Path.PREREQUISITE_IMAGE.value, Path.SEG_PREREQUISITE_IMAGE.value,
                             SD_DOMAIN, PREPROCESSOR_RESOLUTION_LIMIT)

        resize_and_save_image(Path.SEG_PREREQUISITE_IMAGE.value, Path.SEG_PREREQUISITE_IMAGE.value, height)
        Room.save_windows_mask(Path.SEG_PREREQUISITE_IMAGE.value, Path.WINDOWS_MASK_INPAINTING_IMAGE.value)

        if Config.DO_POSTPROCESSING.value and Config.UI.value == "comfyui":
            processor = PostProcessor()
            processor.execute()

    def calculate_kitchen_parameters(self, all_sides, camera_angles_rad: tuple):
        if len(all_sides) > 0:
            side = all_sides.pop(0)
        else:
            return None

        from stage.furniture.KitchenSet import KitchenSet

        # Получаем коэффициенты пикселей на метр
        ratio_x, ratio_y = self.floor_layout.get_pixels_per_meter_ratio()
        pixels_dict = self.floor_layout.get_pixels_dict()

        # Рассчитываем разницу в пикселях и смещение для модели кухни
        middle_point = side.get_middle_point()
        pixel_diff = -1 * (middle_point[0] - pixels_dict['camera'][0]), middle_point[1] - pixels_dict['camera'][1]
        kitchen_offset_x_y = self.floor_layout.calculate_offset_from_pixel_diff(pixel_diff, (ratio_x, ratio_y))

        pitch_rad, roll_rad = camera_angles_rad
        kitchen_set = KitchenSet()

        # Рассчитываем угол стены (yaw angle)
        yaw_angle = side.calculate_wall_angle()

        # Получаем параметры рендеринга из базового метода calculate_rendering_parameters
        render_parameters = kitchen_set.calculate_rendering_parameters(
            self, kitchen_offset_x_y, yaw_angle, (roll_rad, pitch_rad)
        )

        return render_parameters

    def calculate_table_parameters(self, camera_angles_rad: tuple):
        from stage.furniture.KitchenTableWithChairs import KitchenTableWithChairs

        # Получаем площадь комнаты
        area = self.floor_layout.estimate_area_from_floor_layout()
        print(area, "AREA in m2")

        # Используем метод для нахождения сторон и вычисляем размеры комнаты
        all_sides = self.floor_layout.find_all_sides_sorted_by_length()
        if not all_sides:
            return None  # Если нет сторон, возвращаем None

        # Предполагаем, что комнаты имеют прямоугольную форму
        room_width = max(
            side.calculate_wall_length(self.floor_layout.ratio_x, self.floor_layout.ratio_y) for side in all_sides)
        room_height = area / room_width if room_width > 0 else 0

        # Вычисляем центр комнаты
        center_x = room_width // 2
        center_y = room_height // 2

        # Получаем угол
        placement_info = KitchenTableWithChairs.find_placement_pixel(self.floor_layout.output_image_path)

        if not placement_info:
            return None  # Если нет доступных пикселей, возвращаем None

        # Сначала выбираем пиксель, ближе к центру комнаты
        placement_candidates = [
            (chosen_pixel, yaw_angle) for chosen_pixel, yaw_angle in placement_info
            if abs(chosen_pixel[0] - center_x) < room_width // 4 and abs(chosen_pixel[1] - center_y) < room_height // 4
        ]

        # Если нашли подходящие кандидаты, выбираем один из них
        if placement_candidates:
            (chosen_pixel, yaw_angle) = placement_candidates[np.random.randint(len(placement_candidates))]
        else:
            # Если нет кандидатов в центре, выбираем случайный пиксель
            (chosen_pixel, yaw_angle) = placement_info[np.random.randint(len(placement_info))]

        # Получаем коэффициенты пикселей на метр
        ratio_x, ratio_y = self.floor_layout.get_pixels_per_meter_ratio()
        pixels_dict = self.floor_layout.get_pixels_dict()

        # Рассчитываем разницу в пикселях и смещение для модели стола
        pixel_diff = -1 * (chosen_pixel[0] - pixels_dict['camera'][0]), chosen_pixel[1] - pixels_dict['camera'][1]
        table_offset_x_y = self.floor_layout.calculate_offset_from_pixel_diff(pixel_diff, (ratio_x, ratio_y))

        # Убедитесь, что передаете правильные значения
        pitch_rad, roll_rad = camera_angles_rad  # Убедитесь, что это действительно углы в радианах
        table = KitchenTableWithChairs()

        # Получаем параметры рендеринга для стола
        render_parameters = table.calculate_rendering_parameters(self, table_offset_x_y, yaw_angle,
                                                                 (roll_rad, pitch_rad))  # Обратите внимание на порядок

        return render_parameters

    def calculate_plant_parameters(self, camera_angles_rad: tuple):

         from stage.furniture.Plant import Plant

         ratio_x, ratio_y = self.floor_layout.get_pixels_per_meter_ratio()
         pixels_dict = self.floor_layout.get_pixels_dict()

         plant_pixels = Plant.find_floor_layout_placement_pixels(self.floor_layout.output_image_path)
         random_index = random.randint(0, len(plant_pixels) - 1)
         plant_point = plant_pixels[random_index]

         pixel_diff = -1 * (plant_point[0] - pixels_dict['camera'][0]), plant_point[1] - pixels_dict['camera'][1]
         plant_offset_x_y = self.floor_layout.calculate_offset_from_pixel_diff(pixel_diff, (ratio_x, ratio_y))

         pitch_rad, roll_rad = camera_angles_rad
         plant = Plant()
         yaw_angle = 0
         render_parameters = plant.calculate_rendering_parameters(self, plant_offset_x_y, yaw_angle, (roll_rad, pitch_rad))
         return render_parameters
