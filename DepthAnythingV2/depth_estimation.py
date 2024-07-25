import argparse
import time

from PIL import Image
import torchvision.transforms as transforms
import torch
import gc
import numpy as np
from tools import get_image_size
import open3d as o3d
import cv2
from constants import Path

depth_npy_path = Path.DEPTH_IMAGE.value
depth_ply_path = Path.PLY_SPACE.value

def image_pixel_to_3d(x, y, image_path, depth_npy_path=depth_npy_path):
    w, h = get_image_size(image_path)
    return transform_to_blender_xyz(*pixel_to_3d(x, y, w, h, depth_npy_path))

def image_pixels_to_3d(image_path, output_path, depth_npy_path=depth_npy_path):
    # WARNING! DOnt forget to use transform_to_blender_xyz
    pixel_coords_3d = get_pixel_3d_coords(image_path, depth_npy_path)
    with open(output_path, "w") as f:
        for coord in pixel_coords_3d:
            f.write(f"{coord[0]},{coord[1]},{coord[2]}\n")

def image_pixel_list_to_3d(image_path, pixels_coordinates: list[list[int,int]], depth_npy_path=depth_npy_path):
    points_3d = []
    w, h = get_image_size(image_path)
    for x, y in pixels_coordinates:
        point_3d = transform_to_blender_xyz(*pixel_to_3d(x, y, w, h, depth_npy_path))
        points_3d.append(point_3d)
    return points_3d


def get_pixel_3d_coords(image_path, depth_npy_path):
    """
    Get 3D coordinates for each pixel in the image.

    Args:
        image_path: path to the image which was used as input
        depth_npy_path: Depth map in .npy format

    Returns:
        List of 3D coordinates for each pixel.
    """
    w, h = get_image_size(image_path)

    # Create arrays to store 3D coordinates
    pixel_coords_3d = []

    for y in range(0, h, 2): # TODO WARNING! Iterating each 10th pixel only to speed up the debug
        for x in range(0, w, 5):
            # Calculate 3D coordinates for each pixel
            pixel_3d = transform_to_blender_xyz(*pixel_to_3d(x, y, w, h, depth_npy_path))
            print(f"Iterating the image: {x, y} -> {pixel_3d}")
            pixel_coords_3d.append(pixel_3d)

    return pixel_coords_3d


def transform_to_blender_xyz(x, y, z):  # TODO test it and visualize the whole depth estimation
    # 1. Invert the y
    # 2. Swap the z and y
    # 3. Invert x
    return -x, z, y


def pixel_to_3d(x, y, w, h, depth_npy_path):
    """
    Args:
        x: x coordinate of the pixel
        y: y coordinate of the pixel
        w, h: original width and height of the image which was used as input
        depth_npy_path: path to the output created by image_pixels_to_depth()

    Returns:
        X_3D, Y_3D, Z_3D: 3D coordinates of pixel
    """
    FX = w * 0.6
    FY = h * 0.9

    depth_image = np.load(depth_npy_path)
    resized_pred = Image.fromarray(depth_image).resize((w, h), Image.NEAREST)

    Z_depth = np.array(resized_pred)[y, x]
    X_3D = (x - w / 2) * Z_depth / FX
    Y_3D = (y - h / 2) * Z_depth / FY
    Z_3D = Z_depth
    X_3D *= -1
    Y_3D *= -1
    return X_3D, Y_3D, Z_3D

def image_pixels_to_point_cloud(image_path, depth_npy_path=depth_npy_path, depth_ply_path=depth_ply_path):
    from DepthAnythingV2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
    # Determine the device to use (CUDA, MPS, or CPU)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Model configuration based on the chosen encoder
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl'
    max_depth = 20
    load_from = Path.DEPTH_CHECKPOINT.value
    height_limit = Path.IMAGE_HEIGHT_LIMIT.value

    # Initialize the DepthAnythingV2 model with the specified configuration
    depth_anything = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    depth_anything.load_state_dict(torch.load(load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Get the list of image files to process
    filenames = [image_path]

    # Process each image file
    for k, filename in enumerate(filenames):
        print(f'Processing {k + 1}/{len(filenames)}: {filename}')

        # Load the image
        color_image = Image.open(filename).convert('RGB')
        width, height = color_image.size

        # Read the image using OpenCV
        image = cv2.imread(filename)
        infer_height = height if height < height_limit else height_limit
        pred = depth_anything.infer_image(image, infer_height)

        # Resize depth prediction to match the original image size
        resized_pred = Image.fromarray(pred).resize((width, height), Image.NEAREST)
        np.save(depth_npy_path, resized_pred)

        # Generate mesh grid and calculate point cloud coordinates
        FX = width * 0.6
        FY = height * 0.9
        focal_length_x, focal_length_y = (FX, FY)
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - width / 2) / focal_length_x
        y = (y - height / 2) / focal_length_y
        z = np.array(resized_pred)
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        colors = np.array(color_image).reshape(-1, 3) / 255.0

        # Create the point cloud and save it to the output directory
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(depth_ply_path,
                                 pcd)


def rotate_3d_points(input_fname, output_fname, pitch_rad, roll_rad): # We rotate them to restore the original global coordinates which were moved due to camera rotation
    # Read points from the .txt file
    points = np.genfromtxt(input_fname, delimiter=',')

    # Define rotation matrices for roll and pitch
    def rotation_matrix_x(theta):
        return np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])

    def rotation_matrix_y(theta):
        return np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

    rotated_points = points.dot(rotation_matrix_x(pitch_rad).T).dot(rotation_matrix_y(roll_rad).T)
    with open(output_fname, "w") as f:
        for coord in rotated_points:
            f.write(f"{coord[0]},{coord[1]},{coord[2]}\n")
    return rotated_points


def rotate_3d_point(point: tuple[float, float, float], pitch_rad, roll_rad):
    def rotation_matrix_x(theta):
        return np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])

    def rotation_matrix_y(theta):
        return np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

    point = np.array(point)
    rotated_point = point.dot(rotation_matrix_x(pitch_rad).T).dot(rotation_matrix_y(roll_rad).T)
    return rotated_point