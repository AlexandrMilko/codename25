import base64
import math
import os.path
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image

from disruptor.stage import Room
from sklearn.cluster import KMeans
from disruptor.tools import get_filename_without_extension, create_directory_if_not_exists


class FurniturePiece:
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


class Bed(FurniturePiece):
    # We use it to scale the model to metric units
    scale = 0.01, 0.01, 0.01
    # We use it to compensate the angle if the model is originally rotated in a wrong way
    default_angles = 0, 0, 90

    def __init__(self, model_path='3Ds/bedroom/bed.obj'):
        super().__init__(model_path)

    @staticmethod
    def find_placement_pixel(wall_mask_path) -> list[list[
        int]]:  # list[list[int]] We return list of coordinates, because some of the furniture pieces, like curtains have numerous copies in the room
        wall_mask = cv2.imread(wall_mask_path, cv2.IMREAD_GRAYSCALE)
        wall_contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate offset from bed centroid to wall centroid
        wall_centroid = np.mean(wall_contours[0], axis=0)[0]
        pixel_x = wall_centroid[0]
        pixel_y = wall_centroid[1]

        return [[int(pixel_x), int(pixel_y)]]

    def calculate_rendering_parameters(self, room, placement_pixel: tuple[int, int],
                                       yaw_angle: float,
                                       camera_angles_rad: tuple[float, float], current_user_id):
        from math import radians
        roll, pitch = camera_angles_rad
        default_angles = self.get_default_angles()

        obj_offsets = room.infer_3d(placement_pixel, pitch,
                                    roll)  # We set negative rotation to compensate
        obj_angles = radians(default_angles[0]), radians(default_angles[1]), radians(
            default_angles[2] + yaw_angle)  # In blender, yaw angle is around z axis. z axis is to the top
        obj_scale = self.get_scale()
        # We set opposite
        camera_angles = radians(
            90) + pitch, -roll, 0  # We add 90 to the pitch, because originally camera is rotated pointing downwards in Blender
        print("Started estimating camera height")
        camera_height = room.estimate_camera_height((pitch, roll), current_user_id)
        print(f"Camera height: {camera_height}")
        camera_location = 0, 0, camera_height
        obj_offsets_floor = obj_offsets.copy()
        obj_offsets_floor[2] = 0

        print("Bed coords")
        print(obj_offsets, "obj_offsets")
        print(obj_offsets_floor, "obj_offsets for blender with floor z axis")
        print(obj_angles, "obj_angles")
        print(yaw_angle, "yaw_angle")
        print(obj_scale, "obj_scale")
        print(camera_angles, "camera_angles")
        print(camera_location, "camera_location")
        print(self.model_path, "model_path")

        params = {
            'obj_offsets': tuple(obj_offsets_floor), # Converting to tuple in case we use ndarrays somewhere which are not JSON serializable
            'obj_angles': tuple(obj_angles),
            'obj_scale': tuple(obj_scale),
            'camera_angles': tuple(camera_angles),
            'camera_location': tuple(camera_location),
            'model_path': self.model_path
        }

        return params

class Curtain(FurniturePiece):
    scale = 0.005, 0.01, 0.01
    # We use it to compensate the angle if the model is originally rotated in a wrong way
    default_angles = 0, 0, 0

    def __init__(self, model_path='3Ds/other/curtain.obj'):
        super().__init__(model_path)

    @staticmethod
    def find_perspective_angle(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        angle_radians = math.atan2(dy, dx)
        angle_degrees = math.degrees(angle_radians)
        if angle_degrees < 0:
            angle_degrees += 360
        return angle_degrees

    @staticmethod
    def find_placement_pixel(window_mask_path: str) -> list[list[tuple[int, int]]]: # We can have many windows, each of them can have 2 points, each point has 2 coords
        """

        Args:
            window_mask_path: путь маски окна

        Returns:
            list[int]: массив коoрдинат точек гардин в формате [[(x1, y1), (x2, y2)]]
                                                                    |          |
                                                            левая точка    правая точка
                                                                    |___________|
                                                                          |
                                                                   гардина целиком
        """
        img = cv2.imread(window_mask_path)  # Замените на ваш путь к файлу

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(thresh, kernel, iterations=1)
        img = cv2.dilate(erosion, kernel, iterations=1)
        # cv2.imshow('gray', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        number_of_windows = Room.Room.find_number_of_windows(window_mask_path)

        corners = cv2.goodFeaturesToTrack(img, number_of_windows * 4, 0.01, 40)
        corners = np.int0(corners)

        # Кластеризация точек
        kmeans = KMeans(n_clusters=number_of_windows)
        points = np.array([i.ravel() for i in corners])
        kmeans.fit(points)
        labels = kmeans.labels_
        saved_points = []

        # Определение и визуализация верхних точек каждого кластера
        for i in range(number_of_windows):
            cluster_points = points[labels == i]
            # Сортировка по Y и выбор двух верхних точек
            top_points = cluster_points[np.argsort(cluster_points[:, 1])][:2]
            for point in top_points:
                saved_points.append(point)

        final_points = []

        for i in range(0, len(saved_points), 2):
            top_left_point = saved_points[i]
            top_right_point = saved_points[i + 1]
            x1, y1 = top_left_point[0], top_left_point[1]
            x2, y2 = top_right_point[0], top_right_point[1]

            # Вычисление угла в радианах
            angle_radians = math.radians(Curtain.find_perspective_angle(x1, y1, x2, y2))

            # Вычисление координат точек слева и справа от верхних углов
            right_top_point = (
                int(top_left_point[0] - 20 * math.cos(angle_radians)), -10 + int(top_left_point[1] - 130 * math.sin(angle_radians)))
            left_top_point = (
                int(top_right_point[0] + 20 * math.cos(angle_radians)), -10 + int(top_right_point[1] - 20 * math.sin(angle_radians)))
            point = [left_top_point, right_top_point]
            final_points.append(point)
            print("Координаты точек слева и справа от верхних углов:")
            print(left_top_point, right_top_point)
            # cv2.circle(img, right_top_point, 50, (0, 255, 0), -1)  # зеленая точка - справа
            # cv2.circle(img, left_top_point, 50, (0, 0, 255), -1)   # красная точка - слева
        return final_points
        # cv2.imshow('Image with Points', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def calculate_rendering_parameters(self, room, placement_pixel: tuple[int, int],
                                       yaw_angle: float,
                                       camera_angles_rad: tuple[float, float], current_user_id):
        from math import radians
        roll, pitch = camera_angles_rad
        default_angles = self.get_default_angles()

        obj_offsets = room.infer_3d(placement_pixel, pitch,
                                    roll)  # We set negative rotation to compensate
        obj_angles = radians(default_angles[0]), radians(default_angles[1]), radians(
            default_angles[2] + yaw_angle)  # In blender, yaw angle is around z axis. z axis is to the top
        obj_scale = self.get_scale()
        # We set opposite
        camera_angles = radians(
            90) + pitch, -roll, 0  # We add 90 to the pitch, because originally camera is rotated pointing downwards in Blender
        # TODO Perform camera height estimation not here, but in stage() function to save computing power
        camera_height = room.estimate_camera_height((pitch, roll), current_user_id)
        camera_location = 0, 0, camera_height

        print("Curtain coords")
        print(obj_offsets, "obj_offsets")
        print(obj_angles, "obj_angles")
        print(yaw_angle, "yaw_angle")
        print(obj_scale, "obj_scale")
        print(camera_angles, "camera_angles")
        print(camera_location, "camera_location")
        print(self.model_path, "model_path")

        params = {
            'obj_offsets': tuple(obj_offsets), # Converting to tuple in case we use ndarrays somewhere which are not JSON serializable
            'obj_angles': tuple(obj_angles),
            'obj_scale': tuple(obj_scale),
            'camera_angles': tuple(camera_angles),
            'camera_location': tuple(camera_location),
            'model_path': self.model_path
        }

        return params