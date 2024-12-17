import base64
import math
import os
import subprocess
from io import BytesIO

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from pxr import Usd, UsdGeom
from sklearn.cluster import DBSCAN

from constants import Path
from lang_segment_anything.app import predict


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


def get_model_dimensions(model_path: str) -> dict:
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


def get_image_bytes(image_path):
    # Open the image with PIL
    img = Image.open(image_path).convert("RGB")  # Convert to RGB for consistent encoding
    # Encode the image as PNG
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    # Get the binary data
    image_bytes = buffer.getvalue()
    return image_bytes


def segment_lang_sam(image_path, output_path):
    inputs = {
        "sam_type": "sam2.1_hiera_small",
        "box_threshold": 0.3,
        "text_threshold": 0.25,
        "text_prompt": "doorway",
        "image_bytes": get_image_bytes(image_path),
    }
    output = predict(inputs)
    output_image = output["output_image"]
    output_image.save(output_path, format="PNG")
    return output["boxes"]


def substract_bbox_from_mask(input_mask_path, output_mask_path, bounding_boxes):
    # Open the image and convert it to grayscale
    image = Image.open(input_mask_path).convert("L")

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Process each bounding box
    for box in bounding_boxes:
        x1, y1, x2, y2 = box

        # Ensure coordinates are within image bounds
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, image_array.shape[1]), min(y2, image_array.shape[0])

        # Set the pixels in the bounding box to black (0 in grayscale)
        image_array[y1:y2, x1:x2] = 0

    # Save the modified image
    result_image = Image.fromarray(image_array)
    result_image.save(output_mask_path)


def find_middles_of_redundant_walls(floor_mask_path, doorway_bboxes,
                                    intersection_mask_path=Path.REDUNDANT_WALLS_ON_FLOOR_MASK_DEBUG_IMAGE.value):
    def intersect_bbox_with_mask(input_mask_path, bounding_boxes, output_mask_path):
        """
        Process the mask image by keeping only the pixels that fall within the given bounding boxes.

        Args:
            input_mask_path (str): Path to the input mask image (PNG format).
            output_mask_path (str): Path to save the processed mask image (PNG format).
            bounding_boxes (list of tuples): List of bounding boxes in the format [(x1, y1, x2, y2), ...].

        Returns:
            list: A one-dimensional array of bottom-left and bottom-right pixel coordinates [(x, y), ...].
        """
        # Open the image and convert it to grayscale
        image = Image.open(input_mask_path).convert("L")

        # Convert the image to a NumPy array
        image_array = np.array(image)

        # Create a blank mask (all black)
        intersected_mask = np.zeros_like(image_array, dtype=np.uint8)

        # Identify white pixels (255 in grayscale)
        white_pixel_mask = image_array == 255

        # Initialize a list to store bottom-left and bottom-right pixels
        bottom_pixels = []

        # Process each bounding box
        for box in bounding_boxes:
            x1, y1, x2, y2 = box

            # Ensure coordinates are within image bounds
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, image_array.shape[1]), min(y2, image_array.shape[0])

            # Extract the region of interest (ROI)
            roi = white_pixel_mask[y1:y2, x1:x2]

            # Check if any white pixels are in the ROI
            if np.any(roi):
                # Add bottom-left and bottom-right coordinates
                bottom_pixels.append([(x1, y2 - 1), (x2 - 1, y2 - 1)])

            # Update the intersected mask for the current bounding box
            intersected_mask[y1:y2, x1:x2] = np.where(roi, 255, intersected_mask[y1:y2, x1:x2])

        # Save the modified image
        result_image = Image.fromarray(intersected_mask)
        result_image.save(output_mask_path)

        return bottom_pixels

    def remove_noise_from_mask(mask_path, output_path):
        # Load the mask image
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Simplify contours using approxPolyDP
        epsilon_factor = 0.01  # Adjust for more or less simplification
        simplified_contours = [cv2.approxPolyDP(contour, epsilon_factor * cv2.arcLength(contour, True), True) for
                               contour in
                               contours]

        # Create a new blank mask and draw simplified contours
        simplified_mask = np.zeros_like(mask_cleaned)
        cv2.drawContours(simplified_mask, simplified_contours, -1, 255, thickness=cv2.FILLED)

        cv2.imwrite(output_path, simplified_mask)

    def find_corners_in_mask(mask_path):
        """
        Find corners in a black-and-white mask using the Harris corner detection algorithm and visualize them.

        Args:
            mask_path (str): Path to the black-and-white mask image (PNG format).
            debug_path (str): Path to save the visualization of detected corners (PNG format).

        Returns:
            list: A list of corner center coordinates [(x, y), ...].
        """
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Detect corners using Harris corner detection
        corners = cv2.cornerHarris(np.float32(mask), blockSize=10, ksize=15, k=0.15)

        # Threshold for corner detection
        threshold = 0.01 * corners.max()

        # Get coordinates of corners
        corner_coords = np.argwhere(corners > threshold)

        # Cluster corner points using DBSCAN
        clustering = DBSCAN(eps=10, min_samples=2).fit(corner_coords)
        labels = clustering.labels_

        # Calculate centers of clusters
        corner_centers = []
        for label in set(labels):
            if label == -1:  # Ignore noise points
                continue
            cluster_points = corner_coords[labels == label]
            center_y, center_x = np.mean(cluster_points, axis=0)
            corner_centers.append((int(center_x), int(center_y)))

        return corner_centers

    def group_pixels_by_contour(image_path, pixels):
        """
        Groups pixels based on their proximity to the contours in the image.

        :param image_path: The path to the image file.
        :param pixels: List or array of pixel coordinates to be grouped based on proximity to contours.
        :return: A list of arrays, where each array contains pixels closest to a particular contour.
        """
        # Read the image
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to get a binary image
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Function to calculate the distance from a point to a contour
        def distance_to_contour(point, contour):
            min_distance = float('inf')  # Initialize with a large value
            for contour_point in contour:
                # Calculate the Euclidean distance between the point and the contour point
                distance = np.linalg.norm(contour_point[0] - point)
                min_distance = min(min_distance, distance)
            return min_distance

        # Initialize an array to store the closest contour for each pixel
        grouped_pixels = {i: [] for i in range(len(contours))}

        # Assign each pixel to the closest contour
        for pixel in pixels:
            distances = [distance_to_contour(pixel, contour) for contour in contours]
            closest_contour = np.argmin(distances)
            grouped_pixels[closest_contour].append(pixel)

        # Convert grouped_pixels to a 2D array
        result = [np.array(grouped_pixels[i]) for i in range(len(contours))]

        for i, group in enumerate(result):
            print(group)
            # Generate a color (using the step of 50 for B, G, and R channels)
            color = (100 * (i % 5), 100 * ((i + 1) % 5), 100 * ((i + 2) % 5))
            for point in group:
                cv2.circle(image, tuple(point), 10, color, -1)  # Draw the point with the assigned color

            # Save the result image with the points in different colors
        cv2.imwrite(image_path, image)

        return result

    def calculate_average_point(p1, p2):
        """Calculate the average point between two pixels."""
        avg_x = (p1[0] + p2[0]) // 2
        avg_y = (p1[1] + p2[1]) // 2
        return avg_x, avg_y

    def find_middles_of_corners(debug_image_path, pixel_groups):
        """Visualizes average points from pairs of pixels and saves the modified image."""
        # Load the debug image using OpenCV
        image = cv2.imread(debug_image_path)

        middles = []

        # Loop through each group of pixels
        for group in pixel_groups:
            # For each pair of pixels in the group, calculate the average point
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    p1 = group[i]
                    p2 = group[j]

                    # Calculate the average point
                    avg_point = calculate_average_point(p1, p2)
                    middles.append(avg_point)

                    # Draw the average point on the image (using a red color)
                    x, y = avg_point
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  # Ensure within image bounds
                        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)  # Red color (BGR format)

        # Save the modified image back to the specified path
        cv2.imwrite(debug_image_path, image)
        return middles

    intersect_bbox_with_mask(floor_mask_path, doorway_bboxes, intersection_mask_path)
    remove_noise_from_mask(intersection_mask_path, intersection_mask_path)
    corners = find_corners_in_mask(intersection_mask_path)
    groups = group_pixels_by_contour(intersection_mask_path, corners)
    return find_middles_of_corners(intersection_mask_path, groups)


def check_pixels_in_white_area(pixel_array, floor_mask_path):
    """
    Check if any specified pixel is in the white area of the floor_mask.png.

    Parameters:
        pixel_array (list or np.ndarray): Array of pixel coordinates [[x1, y1], [x2, y2], ...].
        floor_mask_path (str): Path to the floor mask image (floor_mask.png).

    Returns:
        bool: True if any pixel is in the white area, False otherwise.
    """
    # Load the floor mask as a binary image (convert to grayscale)
    floor_mask = Image.open(floor_mask_path).convert('L')
    mask_array = np.array(floor_mask)

    # Convert white areas to boolean (255 -> True)
    white_area = mask_array == 255

    # Check each pixel
    for pixel in pixel_array:
        x, y = pixel

        # Ensure the pixel coordinates are within the mask dimensions
        if 0 <= y < white_area.shape[0] and 0 <= x < white_area.shape[1]:
            if white_area[y, x]:  # Check if the pixel is white
                return True

    return False


def bounding_boxes_to_pixels(bounding_boxes):
    """
    Convert a list of bounding boxes to a list of corner pixels.

    Parameters:
        bounding_boxes (list or np.ndarray): Array of bounding boxes in the format [x_min, y_min, x_max, y_max].

    Returns:
        np.ndarray: Array of corner pixels in the format [[x1, y1], [x2, y2], ...].
    """
    # Ensure the input is a NumPy array for easier manipulation
    bounding_boxes = np.array(bounding_boxes)

    # Extract corner points
    top_left = bounding_boxes[:, :2]  # [x_min, y_min]
    bottom_right = bounding_boxes[:, 2:]  # [x_max, y_max]

    # Combine top-left and bottom-right into a single array of pixels
    pixels = np.vstack((top_left, bottom_right))

    return pixels


def euclidean_distance(p1, p2, ratio_x, ratio_y):
    """
    Scaled Euclidean distance between two points p1 and p2 using ratio_x and ratio_y.

    :param p1: Tuple (x1, y1)
    :param p2: Tuple (x2, y2)
    :param ratio_x: Ratio for scaling x-axis
    :param ratio_y: Ratio for scaling y-axis
    :return: Scaled Euclidean distance
    """
    dx = (p2[0] - p1[0]) / ratio_x
    dy = (p2[1] - p1[1]) / ratio_y
    return math.sqrt(dx ** 2 + dy ** 2)