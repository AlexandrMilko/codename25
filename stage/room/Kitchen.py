import json
import random

import numpy as np

from constants import Path, Config
from postprocessing.postProcessing import PostProcessor
from preprocessing.preProcessSegment import ImageSegmentor
from tools import resize_and_save_image
from .Room import Room
from ..furniture.Furniture import Furniture


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

        segment = ImageSegmentor(Path.RENDER_IMAGE.value, Path.SEG_RENDER_IMAGE.value, PREPROCESSOR_RESOLUTION_LIMIT)
        segment.execute()

        resize_and_save_image(Path.SEG_RENDER_IMAGE.value, Path.SEG_RENDER_IMAGE.value, height)
        Room.save_windows_mask(Path.SEG_RENDER_IMAGE.value, Path.WINDOWS_MASK_INPAINTING_IMAGE.value)

        if Config.DO_POSTPROCESSING.value:
            processor = PostProcessor()
            processor.execute()

    def calculate_kitchen_parameters(self, all_sides, camera_angles_rad: tuple):
        from stage.furniture.KitchenSet import KitchenSet
        from tools import get_model_dimensions  # Импорт функции для расчета размеров модели

        if len(all_sides) == 0:
            return None

        side = all_sides.pop(0)

        # Получаем размеры стены
        wall_length = side.length
        wall_height = side.height

        # Доступные модели кухни
        kitchen_models = [
            {'name': Path.KITCHEN_BIG_MODEL.value, **get_model_dimensions(Path.KITCHEN_BIG_MODEL.value)},
            {'name': Path.KITCHEN_SMALL_ONE.value, **get_model_dimensions(Path.KITCHEN_SMALL_ONE.value)},
            {'name': Path.KITCHEN_SMALL_TWO.value, **get_model_dimensions(Path.KITCHEN_SMALL_TWO.value)},
            {'name': Path.KITCHEN_SMALL_THREE.value, **get_model_dimensions(Path.KITCHEN_SMALL_THREE.value)},
        ]

        # Фильтруем подходящие модели
        suitable_models = [
            model for model in kitchen_models
            if model['length'] <= wall_length and model['height'] <= wall_height
        ]

        if not suitable_models:
            return None  # Нет подходящих моделей

        # Выбираем случайную модель из подходящих
        chosen_model = random.choice(suitable_models)

        # Создаем экземпляр KitchenSet с выбранной моделью
        kitchen_set = KitchenSet(chosen_model['name'])

        # Рассчитываем параметры размещения
        ratio_x, ratio_y = self.floor_layout.get_pixels_per_meter_ratio()
        pixels_dict = self.floor_layout.get_pixels_dict()
        middle_point = side.get_middle_point()
        pixel_diff = -1 * (middle_point[0] - pixels_dict['camera'][0]), middle_point[1] - pixels_dict['camera'][1]
        kitchen_offset_x_y = self.floor_layout.calculate_offset_from_pixel_diff(pixel_diff, (ratio_x, ratio_y))

        yaw_angle = side.calculate_wall_angle()
        pitch_rad, roll_rad = camera_angles_rad

        render_parameters = kitchen_set.calculate_rendering_parameters(
            self, kitchen_offset_x_y, yaw_angle, (roll_rad, pitch_rad)
        )
        return render_parameters

    def calculate_table_parameters(self, camera_angles_rad: tuple):
        from stage.furniture.KitchenTableWithChairs import KitchenTableWithChairs
        from tools import get_model_dimensions  # Импорт функции для расчета размеров модели

        # Определяем место для стола
        placement_info = KitchenTableWithChairs.find_placement_pixel(self.floor_layout.output_image_path)

        if not placement_info:
            return None  # Нет подходящих мест для стола

        (chosen_pixel, yaw_angle) = placement_info[0]

        # Получаем размеры доступного пространства
        space_length = self.get_available_space_length()
        space_width = self.get_available_space_width()

        # Доступные модели столов
        table_models = [
            {'name': Path.KITCHEN_TABLE_MODEL_ONE.value, **get_model_dimensions(Path.KITCHEN_TABLE_MODEL_ONE.value)},
            {'name': Path.KITCHEN_TABLE_MODEL_TWO.value, **get_model_dimensions(Path.KITCHEN_TABLE_MODEL_TWO.value)},
        ]

        # Фильтруем подходящие модели
        suitable_models = [
            model for model in table_models
            if model['length'] <= space_length and model['width'] <= space_width
        ]

        if not suitable_models:
            return None  # Нет подходящих моделей

        # Выбираем случайную модель из подходящих
        chosen_model = random.choice(suitable_models)

        # Создаем экземпляр KitchenTableWithChairs с выбранной моделью
        table = KitchenTableWithChairs(chosen_model['name'])

        # Рассчитываем параметры размещения
        ratio_x, ratio_y = self.floor_layout.get_pixels_per_meter_ratio()
        pixels_dict = self.floor_layout.get_pixels_dict()
        pixel_diff = -1 * (chosen_pixel[0] - pixels_dict['camera'][0]), chosen_pixel[1] - pixels_dict['camera'][1]
        table_offset_x_y = self.floor_layout.calculate_offset_from_pixel_diff(pixel_diff, (ratio_x, ratio_y))

        pitch_rad, roll_rad = camera_angles_rad

        render_parameters = table.calculate_rendering_parameters(
            self, table_offset_x_y, yaw_angle, (roll_rad, pitch_rad)
        )
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
