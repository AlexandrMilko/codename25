from stage.Room import Room
from stage.Floor import Floor
from stage.Furniture import Furniture, Bed, Curtain, Plant
from tools import calculate_angle_from_top_view, get_image_size, create_mask_of_size, convert_png_to_mask, overlay_masks
import numpy as np
import os
from math import radians
import time

from PIL import Image
from tools import image_overlay


class Bedroom(Room):

    def stage(self):
        roll, pitch = np.negative(np.degrees(self.find_roll_pitch()))
        print(roll, pitch, "ROLL and PITCH of the CAMERA")
        pitch_rad, roll_rad = radians(pitch), radians(roll)

        # Add time for Garbage Collector
        time.sleep(5)

        from stage.DepthAnything.depth_estimation import image_pixels_to_point_cloud, depth_ply_path, depth_npy_path
        image_pixels_to_point_cloud(self.empty_room_image_path)
        floor_layout_path = 'images/preprocessed/floor_layout.png'
        self.save_floor_layout_image(depth_ply_path, depth_npy_path, floor_layout_path)

        # Add time for Garbage Collector
        time.sleep(5)

        # from stage.DepthAnything.depth_estimation import image_pixels_to_3d, rotate_3d_points
        # image_pixels_to_3d(self.empty_room_image_path, "my_3d_space.txt")
        # rotate_3d_points("my_3d_space.txt", "my_3d_space_rotated.txt", -pitch_rad, -roll_rad)

        # Segment our empty space room. It is used in Room.save_windows_mask
        from tools import get_image_size, run_preprocessor
        width, height = get_image_size(self.empty_room_image_path)
        run_preprocessor("seg_ofade20k", self.empty_room_image_path, "segmented_es.png", height)

        camera_height = self.estimate_camera_height([pitch_rad, roll_rad])

        # Create an empty mask of same size as image
        mask_path = f'images/preprocessed/furniture_mask.png'
        tmp_mask_path = f'images/preprocessed/furniture_piece_mask.png'
        width, height = get_image_size(self.empty_room_image_path)
        empty_mask = create_mask_of_size(width, height)
        print("Saving empty mask to:", mask_path)
        empty_mask.save(mask_path)
        print("Empty mask saved successfully!")

        # Add curtains
        prerequisite_path = f'images/preprocessed/prerequisite.png'
        curtain = Curtain()
        segmented_es_path = f'images/preprocessed/segmented_es.png'
        Room.save_windows_mask(segmented_es_path, f'images/preprocessed/windows_mask.png')
        pixels_for_placing = curtain.find_placement_pixel(
            f'images/preprocessed/windows_mask.png')
        print(f"CURTAINS placement pixels: {pixels_for_placing}")
        Image.open(self.empty_room_image_path).save(prerequisite_path)
        for window in pixels_for_placing:
            try:
                left_top_point, right_top_point = window
                yaw_angle = calculate_angle_from_top_view(*[self.infer_3d(pixel, pitch_rad, roll_rad) for
                                                            pixel in (left_top_point, right_top_point)])
                for pixel in (left_top_point, right_top_point):
                    render_parameters = curtain.calculate_rendering_parameters(self, pixel, yaw_angle,
                                                                               (roll_rad, pitch_rad))
                    width, height = get_image_size(self.empty_room_image_path)
                    render_parameters['resolution_x'] = width
                    render_parameters['resolution_y'] = height
                    curtains_height = camera_height + render_parameters['obj_offsets'][2]
                    curtains_height_scale = curtains_height / Curtain.default_height
                    render_parameters['obj_scale'] = render_parameters['obj_scale'][0], render_parameters['obj_scale'][1], curtains_height_scale
                    curtain_image = curtain.request_blender_render(render_parameters)
                    curtain_image.save(tmp_mask_path)
                    convert_png_to_mask(tmp_mask_path)
                    overlay_masks(tmp_mask_path, mask_path, mask_path, [0, 0])
                    background_image = Image.open(prerequisite_path)
                    combined_image = image_overlay(curtain_image, background_image)
                    combined_image.save(prerequisite_path)
            except IndexError as e:
                print(f"{e}, we skip adding curtains for a window.")

        # Add time for Garbage Collector
        time.sleep(5)

        # Add plant
        plant = Plant()
        seg_image_path = f'images/preprocessed/segmented_es.png'
        save_path = 'images/preprocessed/floor_mask.png'
        Floor.save_mask(seg_image_path, save_path)
        pixels_for_placing = plant.find_placement_pixel(save_path)
        print(f"PLANT placement pixels: {pixels_for_placing}")
        import random
        random_index = random.randint(0, len(pixels_for_placing) - 1)
        plant_yaw_angle = 0  # We do not rotate plants
        render_parameters = (plant.calculate_rendering_parameters(self, pixels_for_placing[random_index], plant_yaw_angle, (roll_rad, pitch_rad)))
        width, height = get_image_size(self.empty_room_image_path)
        render_parameters['resolution_x'] = width
        render_parameters['resolution_y'] = height
        plant_image = plant.request_blender_render(render_parameters)
        plant_image.save(tmp_mask_path)
        convert_png_to_mask(tmp_mask_path)
        overlay_masks(tmp_mask_path, mask_path, mask_path, [0, 0])
        background_image = Image.open(prerequisite_path)
        combined_image = image_overlay(plant_image, background_image)
        combined_image.save(prerequisite_path)

        # Add time for Garbage Collector
        time.sleep(5)

        # Add Bed
        bed = Bed()
        wall = self.get_biggest_wall()
        render_directory = f'images/preprocessed'
        wall.save_mask(os.path.join(render_directory, 'wall_mask.png'))
        pixel_for_placing = bed.find_placement_pixel(os.path.join(render_directory, 'wall_mask.png'))
        print(f"BED placement pixel: {pixel_for_placing}")
        yaw_angle = wall.find_angle_from_3d(self, pitch_rad, roll_rad)
        render_parameters = (bed.calculate_rendering_parameters(self, pixel_for_placing, yaw_angle, (roll_rad, pitch_rad)))
        width, height = get_image_size(self.empty_room_image_path)
        render_parameters['resolution_x'] = width
        render_parameters['resolution_y'] = height
        bed_image = bed.request_blender_render(render_parameters)
        bed_image.save(tmp_mask_path)
        convert_png_to_mask(tmp_mask_path)
        overlay_masks(tmp_mask_path, mask_path, mask_path, [0, 0])
        background_image = Image.open(prerequisite_path)
        combined_image = image_overlay(bed_image, background_image)
        combined_image.save(prerequisite_path)

        # Create windows mask for staged room
        run_preprocessor("seg_ofade20k", prerequisite_path, "seg_prerequisite.png", height)
        segmented_es_path = f'images/preprocessed/seg_prerequisite.png'
        Room.save_windows_mask(segmented_es_path,
                               f'images/preprocessed/windows_mask_inpainting.png')