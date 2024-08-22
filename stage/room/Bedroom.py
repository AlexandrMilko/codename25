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

        scene_render_parameters['objects'] = [bed_parameters]
        import json
        print(json.dumps(scene_render_parameters, indent = 4))

        # Add curtains
        # curtains_parameters = self.calculate_curtains_parameters(camera_height, (pitch_rad, roll_rad))

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

        middle_point, longest_side_points = self.floor_layout.find_middle_of_longest_side()
        ratio_x, ratio_y = self.floor_layout.get_pixels_per_meter_ratio()
        pixels_dict = self.floor_layout.get_pixels_dict()

        print(middle_point, pixels_dict)
        print(ratio_x, ratio_y, "ratios")

        pixel_diff = -1* (middle_point[0] - pixels_dict['camera'][0][0]), middle_point[1] - pixels_dict['camera'][0][1]
        bed_offset_x_y = self.floor_layout.calculate_offset_from_pixel_diff(pixel_diff, (ratio_x, ratio_y))
        print(bed_offset_x_y, "Bed offset")

        pitch_rad, roll_rad = camera_angles_rad
        bed = Bed()
        print(f"BED floor placement pixel: {middle_point}")
        yaw_angle = self.floor_layout.calculate_wall_angle(middle_point, longest_side_points)
        render_parameters = (
            bed.calculate_rendering_parameters(self, bed_offset_x_y, yaw_angle, (roll_rad, pitch_rad)))
        return render_parameters
