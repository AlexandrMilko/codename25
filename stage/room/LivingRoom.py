from tools import get_image_size, convert_png_to_mask, overlay_masks, run_preprocessor, image_overlay
from constants import Path
from .Room import Room
from PIL import Image
from ..furniture.Furniture import Furniture

class LivingRoom(Room):
    def stage(self):
        camera_height, pitch_rad, roll_rad, height, scene_render_parameters = self.prepare_empty_room_data()

        # Add curtains
        curtains_parameters = self.add_curtains(camera_height, (pitch_rad, roll_rad),
                          Path.FURNITURE_MASK_IMAGE.value,
                          Path.FURNITURE_PIECE_MASK_IMAGE.value,
                          Path.PREREQUISITE_IMAGE.value)

        # TODO change algo for plant with new Kyrylo algorithm
        # self.add_plant((pitch_rad, roll_rad), Path.FURNITURE_MASK_IMAGE.value, Path.FURNITURE_PIECE_MASK_IMAGE.value, Path.PREREQUISITE_IMAGE.value)

        # Add kitchen_table_with_chairs
        sofa_parameters = self.add_sofa_with_table((pitch_rad, roll_rad),
                                 Path.FURNITURE_MASK_IMAGE.value,
                                 Path.FURNITURE_PIECE_MASK_IMAGE.value,
                                 Path.PREREQUISITE_IMAGE.value)

        scene_render_parameters['objects'] = [*curtains_parameters, sofa_parameters]

        import json
        print(json.dumps(scene_render_parameters, indent=4))

        furniture_image = Furniture.request_blender_render(scene_render_parameters)
        furniture_image.save(Path.FURNITURE_PIECE_MASK_IMAGE.value)
        convert_png_to_mask(Path.FURNITURE_PIECE_MASK_IMAGE.value)
        overlay_masks(Path.FURNITURE_PIECE_MASK_IMAGE.value, Path.FURNITURE_MASK_IMAGE.value,
                      Path.FURNITURE_MASK_IMAGE.value)
        background_image = Image.open(Path.PREREQUISITE_IMAGE.value)
        combined_image = image_overlay(furniture_image, background_image)
        combined_image.save(Path.PREREQUISITE_IMAGE.value)

        # Create windows mask for staged room
        run_preprocessor("seg_ofade20k", Path.PREREQUISITE_IMAGE.value, "seg_prerequisite.png", height)
        Room.save_windows_mask(Path.SEG_PREREQUISITE_IMAGE.value, Path.WINDOWS_MASK_INPAINTING_IMAGE.value)

    def add_sofa_with_table(self, camera_angles_rad: tuple, mask_path, tmp_mask_path, prerequisite_path):
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
        # width, height = get_image_size(self.empty_room_image_path)
        # render_parameters['resolution_x'] = width
        # render_parameters['resolution_y'] = height
        return render_parameters
        sofa_image = sofa_with_table.request_blender_render(render_parameters)
        sofa_image.save(tmp_mask_path)
        convert_png_to_mask(tmp_mask_path)
        overlay_masks(tmp_mask_path, mask_path, mask_path)
        background_image = Image.open(prerequisite_path)
        combined_image = image_overlay(sofa_image, background_image)
        combined_image.save(prerequisite_path)
