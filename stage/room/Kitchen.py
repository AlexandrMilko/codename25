from postprocessing.postProcessing import PostProcessor
from preprocessing.preProcessSegment import ImageSegmentor
from constants import Path
from tools import resize_and_save_image
from .Room import Room
from ..furniture.Furniture import Furniture
import os


class Kitchen(Room):
    def stage(self):
        camera_height, pitch_rad, roll_rad, height, scene_render_parameters = self.prepare_empty_room_data()

        # Add curtains
        curtains_parameters = self.calculate_curtains_parameters(camera_height, (pitch_rad, roll_rad))

        # Add plant
        # TODO change algo for plant with new Kyrylo algorithm
        # self.calculate_plant_parameters((pitch_rad, roll_rad))

        # Add kitchen_table_with_chairs
        table_with_chairs_parameters = self.calculate_kitchen_table_with_chairs_parameters((pitch_rad, roll_rad))

        scene_render_parameters['objects'] = [*curtains_parameters, table_with_chairs_parameters]

        import json
        print(json.dumps(scene_render_parameters, indent=4))

        furniture_image = Furniture.request_blender_render(scene_render_parameters)
        Room.process_rendered_image(furniture_image)

        processor = PostProcessor()
        processor.execute()

        PREPROCESSOR_RESOLUTION_LIMIT = 1024 if height > 1024 else height

        segment = ImageSegmentor(Path.PREREQUISITE_IMAGE.value, Path.SEG_PREREQUISITE_IMAGE.value, PREPROCESSOR_RESOLUTION_LIMIT)
        segment.execute()
        resize_and_save_image( Path.SEG_PREREQUISITE_IMAGE.value,
                              Path.SEG_PREREQUISITE_IMAGE.value, height)
        Room.save_windows_mask(Path.SEG_PREREQUISITE_IMAGE.value, Path.WINDOWS_MASK_INPAINTING_IMAGE.value)

        # room = Room(Path.INPUT_IMAGE.value)
        # room.create_floor_layout(pitch_rad, roll_rad)

    def calculate_kitchen_table_with_chairs_parameters(self, camera_angles_rad: tuple):
        from stage.furniture.KitchenTableWithChairs import KitchenTableWithChairs
        from stage.Floor import Floor
        import random
        pitch_rad, roll_rad = camera_angles_rad

        kitchen_table_with_chairs = KitchenTableWithChairs()
        seg_image_path = Path.SEGMENTED_ES_IMAGE.value
        save_path = Path.FLOOR_MASK_IMAGE.value
        Floor.save_mask(seg_image_path, save_path)

        pixels_for_placing = kitchen_table_with_chairs.find_placement_pixel(Path.FLOOR_LAYOUT_IMAGE.value)
        print(f"KitchenTableWithChairs placement pixel: {pixels_for_placing}")
        wall = self.get_biggest_wall()
        wall.save_mask(Path.WALL_MASK_IMAGE.value)
        yaw_angle = Floor.find_angle_from_floor_layout(pitch_rad, roll_rad)
        random_index = random.randint(0, len(pixels_for_placing) - 1)
        render_parameters = (
            kitchen_table_with_chairs.calculate_rendering_parameters(self, pixels_for_placing[random_index], yaw_angle,
                                                                     (roll_rad, pitch_rad)))
        return render_parameters