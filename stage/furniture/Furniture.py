from math import radians
from io import BytesIO
from PIL import Image
import requests
import base64


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

    def calculate_rendering_parameters_without_offsets(self,
                                       yaw_angle: float):
        default_angles = self.get_default_angles()

        obj_angles = radians(default_angles[0]), radians(default_angles[1]), radians(default_angles[2] + yaw_angle)
        obj_scale = self.get_scale()

        params = {
            # Converting to tuple in case we use ndarrays somewhere which are not JSON serializable
            'obj_angles': tuple(obj_angles),
            'obj_scale': tuple(obj_scale),
            'model_path': self.model_path
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
        obj_offsets = room.infer_3d(placement_pixel, pitch,
                                    roll)  # We set negative rotation to compensate

        params = self.calculate_rendering_parameters_without_offsets(yaw_angle)
        params['obj_offsets'] = tuple(obj_offsets)

        return params
