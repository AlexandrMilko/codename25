from postprocessing.postProcessing import PostProcessor
from preprocessing.preProcessSegment import ImageSegmentor
from constants import Path
from .Room import Room
import os
from ..furniture.Furniture import Furniture


class Bedroom(Room):
    def stage(self):
        camera_height, pitch_rad, roll_rad, height, scene_render_parameters = self.prepare_empty_room_data()

        # Add curtains
        curtains_parameters = self.calculate_curtains_parameters(camera_height, (pitch_rad, roll_rad))

        # Add plant
        # TODO change algo for plant with new Kyrylo algorithm
        # self.calculate_plant_parameters((pitch_rad, roll_rad))

        # Add kitchen_table_with_chairs
        bed_parameters = self.calculate_bed_parameters((pitch_rad, roll_rad))

        scene_render_parameters['objects'] = [*curtains_parameters, bed_parameters]

        import json
        print(json.dumps(scene_render_parameters, indent = 4))

        furniture_image = Furniture.request_blender_render(scene_render_parameters)
        Room.process_rendered_image(furniture_image)

        processor = PostProcessor()
        processor.execute()

        # Create windows mask for staged room
        PREPROCESSOR_RESOLUTION_LIMIT = 1024 if height > 1024 else height

        segment = ImageSegmentor(Path.PREREQUISITE_IMAGE.value, os.path.join(Path.APP_DIR.value, Path.SEG_PREREQUISITE_IMAGE.value), PREPROCESSOR_RESOLUTION_LIMIT)
        segment.execute()
        Room.save_windows_mask(Path.SEG_PREREQUISITE_IMAGE.value, Path.WINDOWS_MASK_INPAINTING_IMAGE.value)

    def calculate_bed_parameters(self, camera_angles_rad: tuple):
        from stage.furniture.Bed import Bed
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
        return render_parameters
