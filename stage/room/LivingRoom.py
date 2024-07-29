from postProcessing import ImageProcessor
from tools import run_preprocessor
from constants import Path
from .Room import Room
from ..furniture.Furniture import Furniture


class LivingRoom(Room):
    def stage(self):
        camera_height, pitch_rad, roll_rad, height, scene_render_parameters = self.prepare_empty_room_data()

        # Add curtains
        curtains_parameters = self.calculate_curtains_parameters(camera_height, (pitch_rad, roll_rad))

        # TODO change algo for plant with new Kyrylo algorithm
        # self.calculate_plant_parameters((pitch_rad, roll_rad))

        # Add kitchen_table_with_chairs
        sofa_parameters = self.calculate_sofa_with_table_parameters((pitch_rad, roll_rad))

        scene_render_parameters['objects'] = [*curtains_parameters, sofa_parameters]

        import json
        print(json.dumps(scene_render_parameters, indent=4))

        furniture_image = Furniture.request_blender_render(scene_render_parameters)
        Room.process_rendered_image(furniture_image)

        processor = ImageProcessor()
        processor.execute()

        # Create windows mask for staged room
        run_preprocessor("seg_ofade20k", Path.PREREQUISITE_IMAGE.value, "seg_prerequisite.png", height)
        Room.save_windows_mask(Path.SEG_PREREQUISITE_IMAGE.value, Path.WINDOWS_MASK_INPAINTING_IMAGE.value)

    def calculate_sofa_with_table_parameters(self, camera_angles_rad: tuple):
        from stage.furniture.SofaWithTable import SofaWithTable
        pitch_rad, roll_rad = camera_angles_rad
        sofa_with_table = SofaWithTable()
        wall = self.get_biggest_wall()
        wall.save_mask(Path.WINDOWS_MASK_IMAGE.value)
        pixel_for_placing = sofa_with_table.find_placement_pixel(Path.WINDOWS_MASK_IMAGE.value)
        print(f"SofaWithTable placement pixel: {pixel_for_placing}")
        yaw_angle = wall.find_angle_from_3d(self, pitch_rad, roll_rad)
        render_parameters = (
            sofa_with_table.calculate_rendering_parameters(self, pixel_for_placing, yaw_angle, (roll_rad, pitch_rad)))
        return render_parameters
