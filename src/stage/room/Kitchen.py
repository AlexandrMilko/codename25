from tools import get_image_size, convert_png_to_mask, overlay_masks, run_preprocessor, image_overlay
from stage.room.Room import Room
from constants import Path
from PIL import Image


class Kitchen(Room):
    def stage(self):
        camera_height, pitch_rad, roll_rad, height = self.prepare_empty_room_data()

        # Add curtains
        self.add_curtains(camera_height, (pitch_rad, roll_rad),
                          Path.FURNITURE_MASK_IMAGE.value,
                          Path.FURNITURE_PIECE_MASK_IMAGE.value,
                          Path.PREREQUISITE_IMAGE.value)

        # Add plant
        # TODO change algo for plant with new Kyrylo algorithm
        # self.add_plant((pitch_rad, roll_rad), mask_path, tmp_mask_path, prerequisite_path)

        # Add kitchen_table_with_chairs
        self.add_kitchen_table_with_chairs((pitch_rad, roll_rad),
                                           Path.FURNITURE_MASK_IMAGE.value,
                                           Path.FURNITURE_PIECE_MASK_IMAGE.value,
                                           Path.PREREQUISITE_IMAGE.value)

        run_preprocessor("seg_ofade20k", Path.PREREQUISITE_IMAGE.value, "seg_prerequisite.png", height)
        Room.save_windows_mask(Path.SEG_PREREQUISITE_IMAGE.value, Path.WINDOWS_MASK_INPAINTING_IMAGE.value)

    def add_kitchen_table_with_chairs(self, camera_angles_rad: tuple, mask_path, tmp_mask_path, prerequisite_path):
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
        yaw_angle = wall.find_angle_from_3d(self, pitch_rad, roll_rad)
        random_index = random.randint(0, len(pixels_for_placing) - 1)
        render_parameters = (
            kitchen_table_with_chairs.calculate_rendering_parameters(self, pixels_for_placing[random_index], yaw_angle,
                                                                     (roll_rad, pitch_rad)))
        width, height = get_image_size(self.empty_room_image_path)
        render_parameters['resolution_x'] = width
        render_parameters['resolution_y'] = height
        table_image = kitchen_table_with_chairs.request_blender_render(render_parameters)
        table_image.save(tmp_mask_path)
        convert_png_to_mask(tmp_mask_path)
        overlay_masks(tmp_mask_path, mask_path, mask_path)
        background_image = Image.open(prerequisite_path)
        combined_image = image_overlay(table_image, background_image)
        combined_image.save(prerequisite_path)

        # Create windows mask for staged room
        run_preprocessor("seg_ofade20k", prerequisite_path, "seg_prerequisite.png", height)
        Room.save_windows_mask(Path.SEGMENTED_ES_IMAGE.value, Path.WINDOWS_MASK_INPAINTING_IMAGE.value)
