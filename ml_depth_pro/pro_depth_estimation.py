import os

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from constants import Path, Config
from tools import get_image_size

# Import from ml_depth_pro
import ml_depth_pro.src.depth_pro as depth_pro
from ml_depth_pro.src.depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

def image_pixel_to_3d(x, y, image_path, focallength_px, depth_npy_path=Path.DEPTH_NPY.value):
    w, h = get_image_size(image_path)
    return pixel_to_3d(x, y, w, h, focallength_px, depth_npy_path)


def pixel_to_3d(x, y, w, h, focallength_px, depth_npy_path):
    FX = focallength_px
    FY = focallength_px

    depth_image = np.load(depth_npy_path)
    print(f"Depth image shape: {depth_image.shape}")
    print(f"x: {x}, y: {y}")

    Z_depth = np.array(depth_image)[y, x]
    X_3D = (x - w / 2) * Z_depth / FX
    Y_3D = -1 * (y - h / 2) * Z_depth / FY
    Z_3D = Z_depth
    print("WARNING!!!! THIS IS OUT POINTS!!!!!", X_3D, Z_3D, Y_3D)
    return X_3D, Z_3D, Y_3D

def image_pixels_to_space_and_floor_point_clouds(image_path,
                                                 depth_npy_path=Path.DEPTH_NPY.value,
                                                 depth_ply_path=Path.DEPTH_PLY.value,
                                                 floor_npy_path=Path.FLOOR_NPY.value,
                                                 floor_ply_path=Path.FLOOR_PLY.value):
    # Initialize model from ml_depth_pro
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, transform = depth_pro.create_model_and_transforms(config=DEFAULT_MONODEPTH_CONFIG_DICT, device=DEVICE)

    # Set model to evaluation mode
    model.eval()

    image, _, f_px = depth_pro.load_rgb(image_path)
    image_tensor = transform(image)

    color_image = Image.open(image_path).convert('RGB')

    # Run inference
    prediction = model.infer(image_tensor, f_px=f_px)
    depth = prediction["depth"].detach().cpu().numpy().squeeze()  # Depth in [m].
    focallength_px = prediction["focallength_px"].detach().cpu().item()  # Focal length in pixels.

    # Normalize inverse depth for visualization
    inverse_depth = 1 / depth
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu)

    # Use colormap to visualize depth map
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('turbo')
    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)  # Color-mapped depth

    # Save depth map for debugging purposes
    color_depth_image = Image.fromarray(color_depth)
    depth_png_path = Path.DEPTH_DEBUG_IMAGE.value
    color_depth_image.save(depth_png_path)

    # Load the mask
    mask = Image.open(Path.FLOOR_MASK_IMAGE.value).convert('L')  # Convert mask to grayscale
    mask_array = np.array(mask)  # Convert to numpy array
    # Ensure the mask and depth have the same dimensions
    if mask_array.shape != depth.shape:
        print(mask_array.shape)
        print(depth.shape)
        print("WARNING!!! The mask and depth data must have the same dimensions.")
        print("WARNING!!! IGNORING IT. Resizing mask to image size.")
        mask_resized = mask.resize(depth.shape[::-1], Image.NEAREST)
        mask_array = np.array(mask_resized)

    # Extract depth values corresponding to white pixels in the mask
    white_pixel_indices = np.where(mask_array == 255)
    filtered_depth = depth[white_pixel_indices]

    # Save raw depth as npy
    np.save(depth_npy_path, depth)
    # Save floor npy
    np.save(floor_npy_path, filtered_depth)

    save_space_as_point_cloud(depth_npy_path, depth_ply_path, focallength_px, color_image)
    save_floor_as_point_cloud(floor_npy_path, floor_ply_path, focallength_px, color_image, white_pixel_indices)

    # **Free up memory here**
    # Delete large variables
    del model, image_tensor, prediction, depth, inverse_depth, color_image, mask, mask_array, white_pixel_indices, filtered_depth

    # Empty CUDA memory cache (only if using CUDA)
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    # Force garbage collection
    import gc
    gc.collect()

    return focallength_px

def save_space_as_point_cloud(npy_path, ply_output_path, focallength_px, color_image):
    width, height = color_image.size
    depth = np.load(npy_path)
    # Generate a mesh grid and calculate point cloud coordinates
    FX = focallength_px
    FY = focallength_px  # Assuming same for X and Y, adjust if necessary
    cx = width / 2  # Assuming principal point is at the center
    cy = height / 2

    # Generate meshgrid for pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Adjust pixel coordinates to normalized image space
    x = (u - cx) / FX
    y = (v - cy) / FY
    z = np.ones_like(x)  # Since depth is along the z-axis (optical axis)

    # Scale x and y by the depth
    x = x * depth
    y = y * depth
    z = depth  # z is already the depth

    # Stack to get the point cloud (Nx3)
    points = np.stack((x, z, -y), axis=-1).reshape(-1, 3)

    # Extract the color data
    colors = np.array(color_image).reshape(-1, 3) / 255.0

    # Create the point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save the point cloud as PLY
    o3d.io.write_point_cloud(ply_output_path, pcd)

    # Visualize the result (Optional)
    # o3d.visualization.draw_geometries([pcd])

def save_floor_as_point_cloud(npy_path, ply_output_path, focallength_px, color_image, white_pixel_indices):
    width, height = color_image.size
    filtered_depth = np.load(npy_path)
    # Generate point cloud coordinates based on the mask
    FX = focallength_px  # Use the focal length provided by the model
    FY = focallength_px
    cx = width / 2  # Assuming the principal point is at the center
    cy = height / 2

    # Extract white pixel indices from the mask
    y_indices, x_indices = white_pixel_indices

    # Compute 3D coordinates from pixel positions and depth
    x = (x_indices - cx) / FX  # Normalize x coordinates
    y = (y_indices - cy) / FY  # Normalize y coordinates
    z = filtered_depth  # Depth is z-axis

    # Scale x and y by depth values to get real-world coordinates
    x = x * z
    y = y * z

    # Stack to create the point cloud (Nx3 array)
    points = np.stack((x, z, -y), axis=-1)  # Note: y is z in Open3D (optical axis is z)

    # Extract corresponding color data for the selected points
    color_image_array = np.array(color_image)
    filtered_colors = color_image_array[y_indices, x_indices] / 255.0

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    # Save point cloud to PLY file
    o3d.io.write_point_cloud(ply_output_path, pcd)

    # Optionally, visualize the point cloud
    # o3d.visualization.draw_geometries([pcd])


def rotate_3d_points(input_fname, output_fname, pitch_rad, roll_rad):
    points = np.genfromtxt(input_fname, delimiter=',')

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

    pcd = o3d.io.read_point_cloud(input_path)
    points = np.asarray(pcd.points)

    has_colors = pcd.has_colors()

    if has_colors:
        colors = np.asarray(pcd.colors)

    rotation_matrix = rotation_matrix_x(pitch_rad).dot(rotation_matrix_y(roll_rad))
    rotated_points = points.dot(rotation_matrix.T)

    rotated_pcd = o3d.geometry.PointCloud()
    rotated_pcd.points = o3d.utility.Vector3dVector(rotated_points)

    if has_colors:
        rotated_pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(output_path, rotated_pcd)
