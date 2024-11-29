import base64
import json
import os
import subprocess

import cv2
import numpy as np
import open3d as o3d
import requests
from PIL import Image
from pxr import Usd, UsdGeom


def calculate_pitch_angle(plane_normal):
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    pitch_angle_rad = np.arctan2(plane_normal[1], plane_normal[2])
    pitch_angle_deg = np.degrees(pitch_angle_rad)

    return pitch_angle_deg


def calculate_roll_angle(plane_normal, reference_vector=(1, 0, 0)):
    # Step 1: Normalize the normal vector of the plane and the reference vector
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    reference_vector = reference_vector / np.linalg.norm(reference_vector)

    # Step 2: Project the plane normal onto the XZ plane (remove a Y component)
    plane_normal_proj_xz = np.array([plane_normal[0], 0, plane_normal[2]])

    # Step 3: Normalize the projected vector
    plane_normal_proj_xz = plane_normal_proj_xz / np.linalg.norm(plane_normal_proj_xz)

    # Step 4: Calculate the dot product between the reference vector and the projected normal
    dot_product = np.dot(reference_vector, plane_normal_proj_xz)

    # Step 5: Compute the roll angle using the arc cosine of the dot product
    roll_angle_rad = np.arccos(dot_product)

    # Step 6: Convert radians to degrees (optional)
    roll_angle_deg = np.degrees(roll_angle_rad) - 90

    return roll_angle_deg


def calculate_plane_normal(ply_path):
    # Step 1: Load the point cloud from the .ply file
    point_cloud = o3d.io.read_point_cloud(ply_path)

    # Step 2: Segment the largest plane using RANSAC
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.01,
                                                     ransac_n=3,
                                                     num_iterations=1000)

    # Step 3: Extract the normal and plane equation (a, b, c, d)
    # Plane equation is: ax + by + cz + d = 0
    [a, b, c, d] = plane_model
    plane_normal = np.array([a, b, c])

    # Print the normal of the plane
    print(f"Plane normal: {plane_normal}")

    # Optionally: visualize the point cloud with the segmented plane
    # inlier_cloud = point_cloud.select_by_index(inliers)
    # outlier_cloud = point_cloud.select_by_index(inliers, invert=True)
    #
    # inlier_cloud.paint_uniform_color([1, 0, 0])  # Paint plane points in red
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, coordinate_frame])
    return plane_normal


def get_encoded_image(image_path):
    img = cv2.imread(image_path)
    # Encode into PNG and send to ControlNet
    try:
        retval, bytes = cv2.imencode('.png', img)
    except cv2.error:
        retval, bytes = cv2.imencode('.jpg', img)
    return base64.b64encode(bytes).decode('utf-8')


def run_subprocess(script_path: str, data=''):
    if os.name == 'nt':
        print("This is a Windows system. Running python")
        subprocess.run(['python', script_path, data], check=True, env=os.environ)
    elif os.name == 'posix':
        print("This is a Unix or Linux system. Running python3")
        subprocess.run(['python3', script_path, data], check=True, env=os.environ)


def create_visuals_dir():
    directories = [
        "visuals/3Ds",
        "visuals/images/preprocessed"
    ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def submit_post(url: str, data: dict):
    """
    Submit a POST request to the given URL with the given data.
    """
    return requests.post(url, data=json.dumps(data))


def save_encoded_image(b64_image: str, output_path: str):
    """
    Save the given image to the given output path.
    """
    with open(output_path, "wb") as image_file:
        image_file.write(base64.b64decode(b64_image))


def resize_and_save_image(input_path, output_path, height):
    with Image.open(input_path) as img:
        aspect_ratio = img.width / img.height
        width = int(height * aspect_ratio)
        resized_img = img.resize((width, height), Image.LANCZOS)
        resized_img.save(output_path)


def downscale_image_if_bigger(input_path, output_path, height):
    with Image.open(input_path) as img:
        aspect_ratio = img.width / img.height
        width = int(height * aspect_ratio)
        img.thumbnail((width, height), Image.LANCZOS)
        img.save(output_path)


def get_encoded_image_from_path(image_path):
    img = cv2.imread(image_path)
    # Encode into PNG and send to ControlNet
    try:
        retval, bytes = cv2.imencode('.png', img)
    except cv2.error:
        retval, bytes = cv2.imencode('.jpg', img)
    return base64.b64encode(bytes).decode('utf-8')


def get_image_size(image_path):
    image = Image.open(image_path)
    width, height = image.size
    image.close()
    return width, height


def calculate_angle_from_top_view(point1, point2):
    points3d = np.array([point1, point2])
    # Project points onto the XY plane (ignoring Z-coordinate)
    projected_points = points3d[:, :2]

    # Calculate the angle between the vector formed by the first and last projected points and the positive, negative X-axis
    vec1 = projected_points[-1] - projected_points[0]
    pos_x = np.array([1, 0])  # Positive X-axis
    neg_x = np.array([-1, 0])  # Negative X-axis
    angle_pos = np.arccos(np.dot(vec1, pos_x) / (np.linalg.norm(vec1) * np.linalg.norm(pos_x)))
    angle_neg = np.arccos(np.dot(vec1, neg_x) / (np.linalg.norm(vec1) * np.linalg.norm(neg_x)))

    # Convert angle from radians to degrees
    angle_pos_degrees = np.degrees(angle_pos)
    angle_neg_degrees = np.degrees(angle_neg)

    cross_product_pos = np.cross(vec1, pos_x)
    rotation_direction_pos = np.sign(cross_product_pos)

    cross_product_neg = np.cross(vec1, neg_x)
    rotation_direction_neg = np.sign(cross_product_neg)

    # We take the angle whichever is smaller from two angles: angle with positive and negative x-axis
    if angle_pos_degrees < angle_neg_degrees:
        return -angle_pos_degrees * rotation_direction_pos
    return -angle_neg_degrees * rotation_direction_neg


from pxr import Usd, UsdGeom, Tf


def get_model_dimensions(model_path):
    """
    Читает размеры модели из файла .usdc.
    :param model_path: Путь к файлу .usdc
    :return: Словарь с длиной, шириной и высотой модели
    """
    stage = Usd.Stage.Open(model_path)
    root_prim = stage.GetDefaultPrim()

    # Создаём объект BBoxCache с корректным аргументом includedPurposes
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_],  # Список токенов
        useExtentsHint=True
    )

    # Вычисляем границы модели
    bbox = bbox_cache.ComputeWorldBound(root_prim)
    bbox_range = bbox.GetRange()
    min_point = bbox_range.GetMin()
    max_point = bbox_range.GetMax()

    # Вычисляем размеры
    length = max_point[0] - min_point[0]
    width = max_point[1] - min_point[1]
    height = max_point[2] - min_point[2]

    return {'length': length, 'width': width, 'height': height}
