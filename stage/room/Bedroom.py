import json
import os
from itertools import permutations

from constants import Path, Config
from postprocessing.postProcessing import PostProcessor
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
        models_path = [
            Path.BED_WITH_TABLES_MODEL.value,
            Path.WARDROBE_MODEL.value,
            Path.COMMODE_MODEL.value
        ]
        points = []
        for sides in permuted_sides:
            for i in range(len(sides)):
                if not sides[i] or not models_path[i]:
                    break
                sides[i].chosen_model_path = models_path[i]

            points.append(self.floor_layout.place_models_on_sides(sides))
        print(f'Points: {points}')

        room_size_required = 6
        output_image_paths = []
        for i in range(len(permuted_sides)):
            bed_parameters = self.calculate_bed_parameters(permuted_sides[i], points[i], (pitch_rad, roll_rad))
            if area < room_size_required:
                wardrobe_parameters, commode_parameters = None, None
            else:  # We will add these types of furniture only if the room is bigger than room_size_required
                wardrobe_parameters = self.calculate_wardrobe_parameters(permuted_sides[i], points[i], (pitch_rad, roll_rad))
                commode_parameters = self.calculate_commode_parameters(permuted_sides[i], points[i], (pitch_rad, roll_rad))
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
            file_path = f"{base}{i}{ext}"
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

    def calculate_bed_parameters(self, all_sides, all_points, camera_angles_rad: tuple):
        from stage.furniture.Bed import Bed
        if len(all_sides) == 0 or len(all_points) == 0:
            return None

        side = all_sides.pop(0)
        point = all_points.pop(0)

        ratio_x, ratio_y = self.floor_layout.get_pixels_per_meter_ratio()
        pixels_dict = self.floor_layout.get_pixels_dict()

        print(f"Point: {point}, Pixels Dictionary: {pixels_dict}")
        print(f"Ratios - X: {ratio_x}, Y: {ratio_y}")

        pixel_diff = -1 * (point[0] - pixels_dict['camera'][0]), point[1] - pixels_dict['camera'][1]
        bed_offset_x_y = self.floor_layout.calculate_offset_from_pixel_diff(pixel_diff, (ratio_x, ratio_y))
        print(bed_offset_x_y, "Bed offset")

        pitch_rad, roll_rad = camera_angles_rad

        # number 3 is hardcoded length of model table with chairs
        # bed = Bed(Path.BED_WITH_TABLES_MODEL.value if side.calculate_wall_length(ratio_x, ratio_y) > 3 else Path.BED_MODEL.value)
        bed = Bed(Path.BED_WITH_TABLES_MODEL.value)

        print(f"BED floor placement pixel: {point}")
        yaw_angle = side.calculate_wall_angle(ratio_x, ratio_y)
        print(yaw_angle, "BED yaw angle in degrees")
        render_parameters = (
            bed.calculate_rendering_parameters(self, bed_offset_x_y, yaw_angle, (roll_rad, pitch_rad)))
        return render_parameters

    def calculate_wardrobe_parameters(self, all_sides, all_points, camera_angles_rad: tuple):
        from stage.furniture.Wardrobe import Wardrobe
        if len(all_sides) == 0 or len(all_points) == 0:
            return None

        side = all_sides.pop(0)
        point = all_points.pop(0)

        ratio_x, ratio_y = self.floor_layout.get_pixels_per_meter_ratio()
        pixels_dict = self.floor_layout.get_pixels_dict()

        pixel_diff = -1 * (point[0] - pixels_dict['camera'][0]), point[1] - pixels_dict['camera'][1]
        wardrobe_offset_x_y = self.floor_layout.calculate_offset_from_pixel_diff(pixel_diff, (ratio_x, ratio_y))
        print(wardrobe_offset_x_y, "Bed offset")

        pitch_rad, roll_rad = camera_angles_rad
        wardrobe = Wardrobe()
        yaw_angle = side.calculate_wall_angle(ratio_x, ratio_y)
        render_parameters = (
            wardrobe.calculate_rendering_parameters(self, wardrobe_offset_x_y, yaw_angle, (roll_rad, pitch_rad)))
        return render_parameters

    def calculate_commode_parameters(self, all_sides, all_points, camera_angles_rad: tuple):
        from stage.furniture.Commode import Commode
        if len(all_sides) == 0 or len(all_points) == 0:
            return None

        side = all_sides.pop(0)
        point = all_points.pop(0)

        ratio_x, ratio_y = self.floor_layout.get_pixels_per_meter_ratio()
        pixels_dict = self.floor_layout.get_pixels_dict()

        pixel_diff = -1 * (point[0] - pixels_dict['camera'][0]), point[1] - pixels_dict['camera'][1]
        commode_offset_x_y = self.floor_layout.calculate_offset_from_pixel_diff(pixel_diff, (ratio_x, ratio_y))
        print(commode_offset_x_y, "Bed offset")

        pitch_rad, roll_rad = camera_angles_rad
        commode = Commode()
        yaw_angle = side.calculate_wall_angle(ratio_x, ratio_y)
        render_parameters = (
            commode.calculate_rendering_parameters(self, commode_offset_x_y, yaw_angle, (roll_rad, pitch_rad)))
        return render_parameters