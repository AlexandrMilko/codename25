import random

from postprocessing.postProcessing import PostProcessor
from preprocessing.preProcessSegment import ImageSegmentor
from constants import Path
from tools import resize_and_save_image
from .Room import Room
import os
from ..furniture.Furniture import Furniture
from ..Floor import Floor


class Bedroom(Room):
    def stage(self):
        camera_height, pitch_rad, roll_rad, height, scene_render_parameters = self.prepare_empty_room_data()

        area = self.floor_layout.estimate_area_from_floor_layout()
        print(area, "AREA in m2")

        bed_parameters = self.calculate_bed_parameters((pitch_rad, roll_rad))
        plant_parameters = self.calculate_plant_parameters((pitch_rad, roll_rad))
        curtains_parameters = self.calculate_curtains_parameters(camera_height, (pitch_rad, roll_rad))

        scene_render_parameters['objects'] = [*curtains_parameters, plant_parameters, bed_parameters]
        import json
        print(json.dumps(scene_render_parameters, indent=4))

        # # Add plant
        # # TODO change algo for plant with new Kyrylo algorithm
        # # self.calculate_plant_parameters((pitch_rad, roll_rad))
        #
        # # Add kitchen_table_with_chairs
        # bed_parameters = self.calculate_bed_parameters((pitch_rad, roll_rad))
        #
        # scene_render_parameters['objects'] = [*curtains_parameters, bed_parameters]
        #
        furniture_image = Furniture.request_blender_render(scene_render_parameters)
        Room.process_rendered_image(furniture_image)

        processor = PostProcessor()
        processor.execute()

        # Create windows mask for staged room
        PREPROCESSOR_RESOLUTION_LIMIT = 1024 if height > 1024 else height
        segment = ImageSegmentor(Path.PREREQUISITE_IMAGE.value, Path.SEG_PREREQUISITE_IMAGE.value, PREPROCESSOR_RESOLUTION_LIMIT)
        segment.execute()
        resize_and_save_image(Path.SEG_PREREQUISITE_IMAGE.value,
                              Path.SEG_PREREQUISITE_IMAGE.value, height)
        Room.save_windows_mask(Path.SEG_PREREQUISITE_IMAGE.value, Path.WINDOWS_MASK_INPAINTING_IMAGE.value)

    def calculate_bed_parameters(self, camera_angles_rad: tuple):
        from stage.furniture.Bed import Bed

        ratio_x, ratio_y = self.floor_layout.get_pixels_per_meter_ratio()
        pixels_dict = self.floor_layout.get_pixels_dict()
        # TODO move find_middle_of_longest_side to BED class?
        longest_side = self.floor_layout.find_middle_of_longest_side()

        all_sides = self.floor_layout.find_all_sides_sorted_by_length()
        print(all_sides, "ALL SIDES")

        print(longest_side, pixels_dict)
        print(ratio_x, ratio_y, "ratios")

        middle_point = longest_side.get_middle_point()
        pixel_diff = -1 * (middle_point[0] - pixels_dict['camera'][0][0]), middle_point[1] - pixels_dict['camera'][0][1]
        bed_offset_x_y = self.floor_layout.calculate_offset_from_pixel_diff(pixel_diff, (ratio_x, ratio_y))
        print(bed_offset_x_y, "Bed offset")

        pitch_rad, roll_rad = camera_angles_rad
        bed = Bed()
        print(f"BED floor placement pixel: {middle_point}")
        yaw_angle = longest_side.calculate_wall_angle()
        print(yaw_angle, "BED yaw angle in degrees")
        render_parameters = (
            bed.calculate_rendering_parameters(self, bed_offset_x_y, yaw_angle, (roll_rad, pitch_rad)))
        return render_parameters

    def calculate_plant_parameters(self, camera_angles_rad: tuple):
        from stage.furniture.Plant import Plant
        ratio_x, ratio_y = self.floor_layout.get_pixels_per_meter_ratio()
        pixels_dict = self.floor_layout.get_pixels_dict()

        plant_pixels = Plant.find_floor_layout_placement_pixels(self.floor_layout.output_image_path)
        random_index = random.randint(0, len(plant_pixels) - 1)
        plant_point = plant_pixels[random_index]

        pixel_diff = -1 * (plant_point[0] - pixels_dict['camera'][0][0]), plant_point[1] - pixels_dict['camera'][0][1]
        plant_offset_x_y = self.floor_layout.calculate_offset_from_pixel_diff(pixel_diff, (ratio_x, ratio_y))

        pitch_rad, roll_rad = camera_angles_rad
        plant = Plant()
        yaw_angle = 0
        render_parameters = (
            plant.calculate_rendering_parameters(self, plant_offset_x_y, yaw_angle, (roll_rad, pitch_rad)))
        return render_parameters