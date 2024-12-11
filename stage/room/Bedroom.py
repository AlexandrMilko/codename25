import os
import random
import json
from itertools import permutations

from constants import Path, Config
from postprocessing.postProcessing import PostProcessor
from preprocessing.preProcessSegment import ImageSegmentor
from tools import resize_and_save_image
from .Room import Room
from ..furniture.Furniture import Furniture


class Bedroom(Room):
    def stage(self):
        camera_height, pitch_rad, roll_rad, height, scene_render_parameters = self.prepare_empty_room_data()

        area = self.floor_layout.estimate_area_from_floor_layout()
        print(area, "AREA in m2")

        all_sides = self.floor_layout.find_all_sides()
        print(all_sides, "ALL SIDES")
        permuted_sides = [list(perm) for perm in permutations(all_sides)]
        print(permuted_sides)

        room_size_required = 6
        output_image_paths = []
        for idx, sides in enumerate(permuted_sides):
            bed_parameters = self.calculate_bed_parameters(sides, (pitch_rad, roll_rad))
            if area < room_size_required:
                wardrobe_parameters, commode_parameters = None, None
            else:  # We will add these types of furniture only if the room is bigger than room_size_required
                wardrobe_parameters = self.calculate_wardrobe_parameters(sides, (pitch_rad, roll_rad))
                commode_parameters = self.calculate_commode_parameters(sides, (pitch_rad, roll_rad))
            plant_parameters = self.calculate_plant_parameters((pitch_rad, roll_rad))
            # curtains_parameters = self.calculate_curtains_parameters(camera_height, (pitch_rad, roll_rad))

            scene_render_parameters['objects'] = [
                # *curtains_parameters,
                plant_parameters, bed_parameters,
                wardrobe_parameters, commode_parameters,
            ]
            # After our parameters calculation som of them will be equal to None, we have to remove them
            scene_render_parameters['objects'] = [item for item in scene_render_parameters['objects'] if item is not None]
            print(json.dumps(scene_render_parameters, indent=4))

            base, ext = os.path.splitext(Path.RENDER_IMAGE.value)
            file_path = f"{base}{idx}{ext}"
            output_image_paths.append(file_path)
            scene_render_parameters['render_path'] = file_path

            Furniture.start_blender_render(scene_render_parameters)

            # WARNING! WE DO NOT USE WINDOW MASK ANYMORE. UNLESS YOU WANT TO ADD CURTAINS
            # PREPROCESSOR_RESOLUTION_LIMIT = Config.CONTROLNET_HEIGHT_LIMIT.value if height > Config.CONTROLNET_HEIGHT_LIMIT.value else height
            # ImageSegmentor(Path.RENDER_IMAGE.value, Path.SEG_RENDER_IMAGE.value, PREPROCESSOR_RESOLUTION_LIMIT).execute()
            #
            # resize_and_save_image(Path.SEG_RENDER_IMAGE.value, Path.SEG_RENDER_IMAGE.value, height)
            # Room.save_windows_mask(Path.SEG_RENDER_IMAGE.value, Path.WINDOWS_MASK_INPAINTING_IMAGE.value)

            if Config.DO_POSTPROCESSING.value:
                PostProcessor().execute()
        return output_image_paths

    def calculate_bed_parameters(self, all_sides, camera_angles_rad: tuple):
        from stage.furniture.Bed import Bed
        if len(all_sides) == 0:
            return None

        side = all_sides.pop(0)

        ratio_x, ratio_y = self.floor_layout.get_pixels_per_meter_ratio()
        pixels_dict = self.floor_layout.get_pixels_dict()

        print(side, pixels_dict)
        print(ratio_x, ratio_y, "ratios")

        middle_point = side.get_middle_point()
        pixel_diff = -1 * (middle_point[0] - pixels_dict['camera'][0]), middle_point[1] - pixels_dict['camera'][1]
        bed_offset_x_y = self.floor_layout.calculate_offset_from_pixel_diff(pixel_diff, (ratio_x, ratio_y))
        print(bed_offset_x_y, "Bed offset")

        pitch_rad, roll_rad = camera_angles_rad

        # number 3 is hardcoded length of model table with chairs
        bed = Bed(Path.BED_WITH_TABLES_MODEL.value if side.calculate_wall_length(ratio_x, ratio_y) > 3 else Path.BED_MODEL.value)

        print(f"BED floor placement pixel: {middle_point}")
        yaw_angle = side.calculate_wall_angle(ratio_x, ratio_y)
        print(yaw_angle, "BED yaw angle in degrees")
        render_parameters = (
            bed.calculate_rendering_parameters(self, bed_offset_x_y, yaw_angle, (roll_rad, pitch_rad)))
        return render_parameters

    def calculate_wardrobe_parameters(self, all_sides, camera_angles_rad: tuple):
        if len(all_sides) > 0:
            side = all_sides.pop(0)
        else:
            return None

        from stage.furniture.Wardrobe import Wardrobe

        ratio_x, ratio_y = self.floor_layout.get_pixels_per_meter_ratio()
        pixels_dict = self.floor_layout.get_pixels_dict()

        middle_point = side.get_middle_point()
        pixel_diff = -1 * (middle_point[0] - pixels_dict['camera'][0]), middle_point[1] - pixels_dict['camera'][1]
        wardrobe_offset_x_y = self.floor_layout.calculate_offset_from_pixel_diff(pixel_diff, (ratio_x, ratio_y))
        print(wardrobe_offset_x_y, "Bed offset")

        pitch_rad, roll_rad = camera_angles_rad
        wardrobe = Wardrobe()
        yaw_angle = side.calculate_wall_angle(ratio_x, ratio_y)
        render_parameters = (
            wardrobe.calculate_rendering_parameters(self, wardrobe_offset_x_y, yaw_angle, (roll_rad, pitch_rad)))
        return render_parameters

    def calculate_commode_parameters(self, all_sides, camera_angles_rad: tuple):
        if len(all_sides) > 0:
            side = all_sides.pop(0)
        else:
            return None

        from stage.furniture.Commode import Commode

        ratio_x, ratio_y = self.floor_layout.get_pixels_per_meter_ratio()
        pixels_dict = self.floor_layout.get_pixels_dict()

        middle_point = side.get_middle_point()
        pixel_diff = -1 * (middle_point[0] - pixels_dict['camera'][0]), middle_point[1] - pixels_dict['camera'][1]
        commode_offset_x_y = self.floor_layout.calculate_offset_from_pixel_diff(pixel_diff, (ratio_x, ratio_y))
        print(commode_offset_x_y, "Bed offset")

        pitch_rad, roll_rad = camera_angles_rad
        commode = Commode()
        yaw_angle = side.calculate_wall_angle(ratio_x, ratio_y)
        render_parameters = (
            commode.calculate_rendering_parameters(self, commode_offset_x_y, yaw_angle, (roll_rad, pitch_rad)))
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
        render_parameters = (
            plant.calculate_rendering_parameters(self, plant_offset_x_y, yaw_angle, (roll_rad, pitch_rad)))
        return render_parameters
