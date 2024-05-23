import argparse
import time

from PIL import Image
from disruptor.stage.DepthAnything.zoedepth.models.builder import build_model
from disruptor.stage.DepthAnything.zoedepth.utils.config import get_config
import torchvision.transforms as transforms
import torch
import gc
import numpy as np
from disruptor.tools import get_image_size

depth_npy_path = 'disruptor/stage/DepthAnything/zoedepth/depth.npy'

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


def image_pixels_to_depth(image_path, depth_npy_path=depth_npy_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default='zoedepth', help="Name of the model to test")
    parser.add_argument("-p", "--pretrained_resource", type=str,
                        default='local::./disruptor/stage/DepthAnything/zoedepth/checkpoints/depth_anything_metric_depth_indoor.pt',
                        help="Pretrained resource to use for fetching weights.")

    args = parser.parse_args()

    config = get_config(args.model, "eval")
    config.pretrained_resource = args.pretrained_resource
    model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    try:
        color_image = Image.open(image_path).convert('RGB')
        image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to(
            'cuda' if torch.cuda.is_available() else 'cpu')

        pred = model(image_tensor)
        if isinstance(pred, dict):
            pred = pred.get('metric_depth', pred.get('out'))
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        pred = pred.squeeze().detach().cpu().numpy()
        np.save(depth_npy_path, pred)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

    del model
    del parser
    del color_image
    del image_tensor
    del pred
    gc.collect()
    torch.cuda.empty_cache()
    # Add time for Garbage Collector
    time.sleep(5)


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