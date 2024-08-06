from PIL import Image
import torch
import numpy as np
from tools import get_image_size
import open3d as o3d
import cv2
from constants import Path

depth_npy_path = Path.DEPTH_IMAGE.value
depth_ply_path = Path.PLY_SPACE.value
floor_npy_path = Path.FLOOR_NPY.value
floor_ply_path = Path.FLOOR_PLY.value

def image_pixel_to_3d(x, y, image_path, depth_npy_path=depth_npy_path):
    w, h = get_image_size(image_path)
    return pixel_to_3d(x, y, w, h, depth_npy_path)

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

    Y_depth = np.array(resized_pred)[y, x]
    X_3D = (x - w / 2) * Y_depth / FX
    Z_3D = -1 * (y - h / 2) * Y_depth / FY
    Y_3D = Y_depth
    return X_3D, Z_3D, Y_3D

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
        # WARNING! This processing must be the same as in pixel_to_3d
        FX = width * 0.6
        FY = height * 0.9
        focal_length_x, focal_length_y = (FX, FY)
        x, z = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - width / 2) / focal_length_x
        z = -1 * (z - height / 2) / focal_length_y
        y = np.array(resized_pred)
        points = np.stack((np.multiply(x, y), np.multiply(z, y), y), axis=-1).reshape(-1, 3)
        colors = np.array(color_image).reshape(-1, 3) / 255.0

        # Create the point cloud and save it to the output directory
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(depth_ply_path,
                                 pcd)

def create_floor_point_cloud(image_path, floor_mask_path=Path.FLOOR_MASK_IMAGE.value, depth_npy_path=floor_npy_path, depth_ply_path=floor_ply_path):
    mask = Image.open(floor_mask_path).convert('L')
    mask_array = np.array(mask)

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
        resized_pred = np.array(Image.fromarray(pred).resize((width, height), Image.NEAREST))

        # Ensure that the depth data and mask have the same dimensions
        if mask_array.shape != resized_pred.shape:
            print(mask_array.shape)
            print(resized_pred.shape)
            print("WARNING!!! The mask and depth data must have the same dimensions.")
            print("WARNING!!! IGNORING IT. Resizing mask to image size")
            mask_resized = mask.resize(resized_pred.shape[::-1], Image.NEAREST)
            mask_array = np.array(mask_resized)
            # raise ValueError("The mask and depth data must have the same dimensions.")

        white_pixel_indices = np.where(mask_array == 255)
        filtered_resized_pred = resized_pred[white_pixel_indices]
        np.save(depth_npy_path, filtered_resized_pred)

        # Generate mesh grid and calculate point cloud coordinates
        # WARNING! This processing must be the same as in pixel_to_3d
        FX = width * 0.6
        FY = height * 0.9
        focal_length_x, focal_length_y = (FX, FY)
        y_indices, x_indices = white_pixel_indices
        # x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x_indices - width / 2) / focal_length_x
        z = -1 * (y_indices - height / 2) / focal_length_y
        y = filtered_resized_pred
        points = np.stack((np.multiply(x, y), np.multiply(z, y), y), axis=-1).reshape(-1, 3)
        color_image_array = np.array(color_image)
        filtered_colors = color_image_array[y_indices, x_indices] / 255.0

        # Create the point cloud and save it to the output directory
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
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

def rotate_ply_file_with_colors(input_path: str, output_path: str, pitch_rad: float, roll_rad: float):
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

    # Load the .ply file using Open3D
    pcd = o3d.io.read_point_cloud(input_path)
    points = np.asarray(pcd.points)

    # Check if the point cloud has colors
    has_colors = pcd.has_colors()

    if has_colors:
        colors = np.asarray(pcd.colors)

    # Calculate the combined rotation matrix
    rotation_matrix = rotation_matrix_x(pitch_rad).dot(rotation_matrix_y(roll_rad))

    # Apply the rotation matrix to each point
    rotated_points = points.dot(rotation_matrix.T)

    # Create a new point cloud with rotated points
    rotated_pcd = o3d.geometry.PointCloud()
    rotated_pcd.points = o3d.utility.Vector3dVector(rotated_points)

    # Retain colors if they exist
    if has_colors:
        rotated_pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save the rotated point cloud to the output file
    o3d.io.write_point_cloud(output_path, rotated_pcd)