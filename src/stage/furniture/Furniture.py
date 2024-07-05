import base64
from io import BytesIO

import requests
from PIL import Image

from math import radians


class Furniture:
    scale = 1, 1, 1
    default_angles = 0, 0, 0

    def __init__(self, model_path):
        self.model_path = model_path

    def get_scale(self):
        return self.scale

    def get_default_angles(self):
        return self.default_angles

    @staticmethod
    def request_blender_render(render_parameters):
        # URL for blender_server
        # blender_server has to be on the host system, because it has access to the screen which is likely needed for rendering
        try:
            server_url = 'http://host.docker.internal:5002/render_image'
            # Send the HTTP request to the server
            response = requests.post(server_url, json=render_parameters)
        except requests.exceptions.ConnectionError:
            server_url = 'http://localhost:5002/render_image'
            # Send the HTTP request to the server
            response = requests.post(server_url, json=render_parameters)

        if response.status_code == 200:
            # Decode the base64 encoded image
            encoded_furniture_image = response.json()['image_base64']
            furniture_image = Image.open(BytesIO(base64.b64decode(encoded_furniture_image)))
            return furniture_image
        else:
            print("Error:", response.status_code, response.text)


class FloorFurniture(Furniture):
    def calculate_rendering_parameters(self, room,
                                       placement_pixel: tuple[int, int],
                                       yaw_angle: float,
                                       camera_angles_rad: tuple[float, float]):
        roll, pitch = camera_angles_rad
        default_angles = self.get_default_angles()

        # We set negative rotation to compensate
        obj_offsets = room.pixel_to_3d(*placement_pixel)
        # In blender, yaw angle is around z axis. z axis is to the top
        obj_angles = radians(default_angles[0]), radians(default_angles[1]), radians(default_angles[2] + yaw_angle)
        obj_scale = self.get_scale()
        # We set opposite
        # We add 90 to the pitch, because originally camera is rotated pointing downwards in Blender
        camera_angles = radians(90) - pitch, +roll, 0
        print("Started estimating camera height")
        camera_height = room.estimate_camera_height((pitch, roll))
        print(f"Camera height: {camera_height}")
        camera_location = 0, 0, camera_height

        params = {
            'obj_offsets': tuple(obj_offsets),
            # Converting to tuple in case we use ndarrays somewhere which are not JSON serializable
            'obj_angles': tuple(obj_angles),
            'obj_scale': tuple(obj_scale),
            'camera_angles': tuple(camera_angles),
            'camera_location': tuple(camera_location),
            'model_path': self.model_path
        }

        return params


class HangingFurniture(Furniture):
    def calculate_rendering_parameters(self, room, placement_pixel: tuple[int, int],
                                       yaw_angle: float,
                                       camera_angles_rad: tuple[float, float]):
        roll, pitch = camera_angles_rad
        default_angles = self.get_default_angles()

        obj_offsets = room.infer_3d(placement_pixel, pitch,
                                    roll)  # We set negative rotation to compensate
        obj_angles = radians(default_angles[0]), radians(default_angles[1]), radians(
            default_angles[2] + yaw_angle)  # In blender, yaw angle is around z axis. z axis is to the top
        obj_scale = self.get_scale()
        # We set opposite
        camera_angles = radians(
            90) - pitch, +roll, 0  # We add 90 to the pitch, because originally camera is rotated pointing downwards in Blender
        # TODO Perform camera height estimation not here, but in stage() function to save computing power
        camera_location = 0, 0, 0

        params = {
            'obj_offsets': tuple(obj_offsets),
            # Converting to tuple in case we use ndarrays somewhere which are not JSON serializable
            'obj_angles': tuple(obj_angles),
            'obj_scale': tuple(obj_scale),
            'camera_angles': tuple(camera_angles),
            'camera_location': tuple(camera_location),
            'model_path': self.model_path
        }

        return params
