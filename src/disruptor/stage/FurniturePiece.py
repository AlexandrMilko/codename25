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
    def find_placement_pixel(wall_mask_path) -> tuple[int, int]:
        wall_mask = cv2.imread(wall_mask_path, cv2.IMREAD_GRAYSCALE)
        wall_contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate offset from bed centroid to wall centroid
        wall_centroid = np.mean(wall_contours[0], axis=0)[0]
        pixel_x = wall_centroid[0]
        pixel_y = wall_centroid[1]

        return int(pixel_x), int(pixel_y)

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
            90) - pitch, +roll, 0  # We add 90 to the pitch, because originally camera is rotated pointing downwards in Blender
        print("Started estimating camera height")
        camera_height = room.estimate_camera_height((pitch, roll), current_user_id)
        print(f"Camera height: {camera_height}")
        camera_location = 0, 0, camera_height
        obj_offsets_floor = obj_offsets.copy()
        obj_offsets_floor[2] = 0

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
    scale = 1, 0.3, 1
    # We use it to compensate the angle if the model is originally rotated in a wrong way
    default_angles = 0, 0, 90

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
    def find_placement_pixel(window_mask_path: str) -> list[list[tuple[int, int]]]:
        img = cv2.imread(window_mask_path, cv2.IMREAD_GRAYSCALE)
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(thresh, kernel, iterations=1)
        img = cv2.dilate(erosion, kernel, iterations=1)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        final_points = []
        # img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for contour in contours:
            contour_points = contour[:, 0, :]

            # Находим крайние точки
            top_left = contour_points[np.argmin(contour_points[:, 0] + contour_points[:, 1])]
            top_right = contour_points[np.argmax(contour_points[:, 0] - contour_points[:, 1])]

            print("Координаты угловых точек:")
            print("Верхняя левая:", top_left)
            print("Верхняя правая:", top_right)

            angle_radians = math.radians(Curtain.find_perspective_angle(top_left[0], top_left[1], top_right[0], top_right[1]))

            # Вычисление координат точек слева и справа от верхних углов
            right_top_point = (
                int(top_left[0] - 20 * math.cos(angle_radians)), -10 + int(top_left[1] - 20 * math.sin(angle_radians)))
            left_top_point = (
                int(top_right[0] + 20 * math.cos(angle_radians)),
                -10 + int(top_right[1] + 20 * math.sin(angle_radians)))
            point = [left_top_point, right_top_point]
            final_points.append(point)

            # cv2.circle(img_vis, right_top_point, 5, (0, 0, 255), -1)  # красная точка - верхняя левая
            # cv2.circle(img_vis, left_top_point, 5, (0, 255, 0), -1)  # зеленая точка - верхняя правая

        # cv2.imshow('Image with Points', img_vis)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return final_points

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
            90) - pitch, +roll, 0  # We add 90 to the pitch, because originally camera is rotated pointing downwards in Blender
        # TODO Perform camera height estimation not here, but in stage() function to save computing power
        camera_location = 0, 0, 0

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

class Plant(FurniturePiece):
    # We use it to scale the model to metric units
    scale = 1, 1, 1
    # We use it to compensate the angle if the model is originally rotated in a wrong way
    default_angles = 0, 0, 0

    def __init__(self, model_path='3Ds/other/plant.obj'):
        super().__init__(model_path)

    @staticmethod
    def is_near_border(x, y, margin, width, height):
        return x < margin or x > width - margin or y < margin or y > height - margin

    @staticmethod
    def find_placement_pixel(floor_mask_path: str) -> list[list[int, int]]:
        # Load the image
        img = cv2.imread(floor_mask_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # WARNING useful only when shape has gray borders
        # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

        # Apply erosion to move points inside
        erosion = 40
        kernel = np.ones((erosion, erosion), np.uint8)  # Adjust the kernel size as needed
        eroded = cv2.erode(gray, kernel, iterations=1)

        # Find contours               (thresh,..
        contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]

        # Approximate the contour
        contour_precision = 0.01
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, contour_precision * perimeter, True)

        # Set margin from border
        margin = 10 + erosion
        # Get image dimensions
        height, width = img.shape[:2]

        points = []
        # Draw points and contours
        for point in approx:
            x, y = point[0]
            if not Plant.is_near_border(x, y, margin, width, height):
                points.append(point[0])
                # cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

        # cv2.drawContours(img, [approx], -1, (0, 255, 0))

        # cv2.imshow('Contour Approximation', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return points

    def calculate_rendering_parameters(self, room, placement_pixel: tuple[int, int],
                                       camera_angles_rad: tuple[float, float], current_user_id):
        from math import radians
        roll, pitch = camera_angles_rad
        default_angles = self.get_default_angles()

        obj_offsets = room.infer_3d(placement_pixel, pitch,
                                    roll)  # We set negative rotation to compensate
        obj_angles = radians(default_angles[0]), radians(default_angles[1]), radians(
            default_angles[2])  # In blender, yaw angle is around z axis. z axis is to the top
        obj_scale = self.get_scale()
        # We set opposite
        camera_angles = radians(
            90) - pitch, +roll, 0  # We add 90 to the pitch, because originally camera is rotated pointing downwards in Blender
        print("Started estimating camera height")
        camera_height = room.estimate_camera_height((pitch, roll), current_user_id)
        print(f"Camera height: {camera_height}")
        camera_location = 0, 0, camera_height
        obj_offsets_floor = obj_offsets.copy()
        obj_offsets_floor[2] = 0

        print(obj_offsets, "obj_offsets")
        print(obj_offsets_floor, "obj_offsets for blender with floor z axis")
        print(obj_angles, "obj_angles")
        print(obj_scale, "obj_scale")
        print(camera_angles, "camera_angles")
        print(camera_location, "camera_location")
        print(self.model_path, "model_path")

        params = {
            'obj_offsets': tuple(obj_offsets_floor),
            # Converting to tuple in case we use ndarrays somewhere which are not JSON serializable
            'obj_angles': tuple(obj_angles),
            'obj_scale': tuple(obj_scale),
            'camera_angles': tuple(camera_angles),
            'camera_location': tuple(camera_location),
            'model_path': self.model_path
        }

        return params

class KitchenTableWithChairs(FurniturePiece):
    # We use it to scale the model to metric units
    scale = 0.01, 0.01, 0.01
    # We use it to compensate the angle if the model is originally rotated in a wrong way
    default_angles = 0, 0, 0

    def __init__(self, model_path='3Ds/kitchen/kitchen_table_with_chairs.obj'):
        super().__init__(model_path)

    @staticmethod
    def find_centers(segments):
        centers = []
        for segment in segments:
            M = cv2.moments(segment)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append([cX, cY])

        return centers

    @staticmethod
    def choose_bigger_segments(segments, total_area):
        segments.sort(key=lambda x: x[1], reverse=True)

        chosen_segments = []
        cumulative_area = 0
        # Change value in order to change the number of places where to put the table
        max_area = 0.9 * total_area
        for segment, area in segments:
            if cumulative_area + area <= max_area:
                chosen_segments.append(segment)
                cumulative_area += area
            else:
                break

        return chosen_segments

    @staticmethod
    def find_segments(contours, bottom_point):
        segments = []
        for i in range(len(contours)):
            start_point = tuple(contours[i][0])
            end_point = tuple(contours[(i + 1) % len(contours)][0])
            # (i + 1) % len(approx) Ensuring it wraps around to
            # the first vertex when processing the last vertex

            segment = np.array([
                start_point,  # Left upper point
                end_point,  # Right upper point
                (end_point[0], bottom_point),  # Right bottom point
                (start_point[0], bottom_point)  # Left bottom point
            ], np.int32)

            area = cv2.contourArea(segment)
            segments.append((segment, area))

        return segments

    @staticmethod
    def find_placement_pixel(floor_mask_path: str) -> list[list[int, int]]:
        image = cv2.imread(floor_mask_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]

        total_area = cv2.contourArea(contour)

        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        segments = KitchenTableWithChairs.find_segments(approx, image.shape[0])
        chosen_segments = KitchenTableWithChairs.choose_bigger_segments(segments, total_area)
        centers = KitchenTableWithChairs.find_centers(chosen_segments)

        # # Draw the segments and centers on the original image
        # for segment in chosen_segments:
        #     cv2.polylines(image, [segment], isClosed=True, color=(0, 255, 0), thickness=2)
        # for center in centers:
        #     cv2.circle(image, center, 5, (0, 0, 255), -1)
        #
        # cv2.imshow('Segments and Centers', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return centers

    def calculate_rendering_parameters(self, room, placement_pixel: tuple[int, int], yaw_angle,
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
            90) - pitch, +roll, 0  # We add 90 to the pitch, because originally camera is rotated pointing downwards in Blender
        print("Started estimating camera height")
        camera_height = room.estimate_camera_height((pitch, roll), current_user_id)
        print(f"Camera height: {camera_height}")
        camera_location = 0, 0, camera_height
        obj_offsets_floor = obj_offsets.copy()
        obj_offsets_floor[2] = 0

        print(obj_offsets, "obj_offsets")
        print(obj_offsets_floor, "obj_offsets for blender with floor z axis")
        print(obj_angles, "obj_angles")
        print(yaw_angle, "yaw_angle")
        print(obj_scale, "obj_scale")
        print(camera_angles, "camera_angles")
        print(camera_location, "camera_location")
        print(self.model_path, "model_path")

        params = {
            'obj_offsets': tuple(obj_offsets_floor),
            # Converting to tuple in case we use ndarrays somewhere which are not JSON serializable
            'obj_angles': tuple(obj_angles),
            'obj_scale': tuple(obj_scale),
            'camera_angles': tuple(camera_angles),
            'camera_location': tuple(camera_location),
            'model_path': self.model_path
        }

        return params

class SofaWithTable(FurniturePiece):
    # We use it to scale the model to metric units
    scale = 1, 1, 1
    # We use it to compensate the angle if the model is originally rotated in a wrong way
    default_angles = 0, 0, 0

    def __init__(self, model_path='3Ds/living_room/sofa_with_table.obj'):
        super().__init__(model_path)

    @staticmethod
    def find_placement_pixel(wall_mask_path) -> tuple[int, int]:
        wall_mask = cv2.imread(wall_mask_path, cv2.IMREAD_GRAYSCALE)
        wall_contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate offset from bed centroid to wall centroid
        wall_centroid = np.mean(wall_contours[0], axis=0)[0]
        pixel_x = wall_centroid[0]
        pixel_y = wall_centroid[1]

        return int(pixel_x), int(pixel_y)

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
            90) - pitch, +roll, 0  # We add 90 to the pitch, because originally camera is rotated pointing downwards in Blender
        print("Started estimating camera height")
        camera_height = room.estimate_camera_height((pitch, roll), current_user_id)
        print(f"Camera height: {camera_height}")
        camera_location = 0, 0, camera_height
        obj_offsets_floor = obj_offsets.copy()
        obj_offsets_floor[2] = 0

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
