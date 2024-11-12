import json
import subprocess
from math import radians

from constants import Path, Config
from tools import run_subprocess
import os


class Furniture:
    scale = 1, 1, 1
    default_angles = 0, 0, 0

    def __init__(self, obj_path):
        self.obj_path = obj_path

    def get_scale(self):
        return self.scale

    def get_default_angles(self):
        return self.default_angles

    @staticmethod
    def start_blender_render(render_parameters):
        data = json.dumps({
            'render_path': Path.RENDER_IMAGE.value,
            'blend_file_path': Path.SCENE_FILE.value,
            'render_samples': Config.RENDER_SAMPLES.value,
            'room_point_cloud_path': render_parameters['room_point_cloud_path'],
            'camera_location': render_parameters['camera_location'],
            'camera_angles': render_parameters['camera_angles'],
            'focal_length_px': render_parameters['focal_length_px'],
            'resolution_x': render_parameters['resolution_x'],
            'resolution_y': render_parameters['resolution_y'],
            'objects': render_parameters['objects'],
            'lights': render_parameters['lights']
        })

        run_subprocess(Path.BLENDER_SCRIPT.value, data)

    def calculate_rendering_parameters_without_offsets(self, yaw_angle: float):
        default_angles = self.get_default_angles()

        obj_angles = radians(default_angles[0]), radians(default_angles[1]), radians(default_angles[2] + yaw_angle)
        obj_scale = self.get_scale()

        params = {
            # Converting to tuple in case we use ndarrays somewhere which are not JSON serializable
            'obj_angles': tuple(obj_angles),
            'obj_scale': tuple(obj_scale),
            'obj_path': self.obj_path
        }

        return params


class FloorFurniture(Furniture):
    def calculate_rendering_parameters(self, room,
                                       obj_offsets_x_y: tuple[float, float],
                                       yaw_angle: float,
                                       camera_angles_rad: tuple[float, float]):
        roll, pitch = camera_angles_rad
        # We set negative rotation to compensate
        obj_offsets = [*obj_offsets_x_y, 0]
        # TODO Perform camera height estimation not here, but in stage() function to save computing power
        print("Started estimating camera height")
        camera_height = room.estimate_camera_height((pitch, roll))
        print(f"Camera height: {camera_height}")
        obj_offsets[2] -= camera_height

        params = self.calculate_rendering_parameters_without_offsets(yaw_angle)
        params['obj_offsets'] = tuple(obj_offsets)

        return params


class HangingFurniture(Furniture):
    def calculate_rendering_parameters(self, room, placement_pixel: tuple[int, int],
                                       yaw_angle: float,
                                       camera_angles_rad: tuple[float, float]):
        roll, pitch = camera_angles_rad
        obj_offsets = room.infer_3d(placement_pixel, pitch,roll)  # We set negative rotation to compensate

        params = self.calculate_rendering_parameters_without_offsets(yaw_angle)
        params['obj_offsets'] = tuple(obj_offsets)

        return params
