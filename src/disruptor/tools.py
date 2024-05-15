import os
import cv2
import base64
import requests
import json
import shutil
from PIL import Image
import numpy as np


def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    # We remove the first 2 points to avoid errors.
    other_pts = [point for point in pts if not np.array_equal(point, rect[0]) and not np.array_equal(point, rect[2])]
    diff = np.diff(other_pts, axis=1)
    # print(diff)
    rect[1] = other_pts[np.argmin(diff)]
    rect[3] = other_pts[np.argmax(diff)]
    # return the ordered coordinates
    # plot_points(rect)
    return rect


def get_filename_without_extension(file_path):
    # Get the base filename from the path
    base_filename = os.path.basename(file_path)
    # Split the filename and extension
    filename_without_extension = os.path.splitext(base_filename)[0]
    return filename_without_extension


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def min_max_scale(value, min_val, max_val):
    if max_val == min_val:
        return 0.5  # Handle division by zero (if min_val == max_val)
    return (value - min_val) / (max_val - min_val)


def move_file(src_path, dest_path):
    try:
        os.rename(src_path, dest_path)
        print(f"File moved successfully from '{src_path}' to '{dest_path}'.")
    except Exception as e:
        print(os.getcwd())
        print(f"Error: {e}")


def copy_file(src_path, dest_path):
    try:
        shutil.copy2(src_path, dest_path)
        print(f"File copied successfully from '{src_path}' to '{dest_path}'.")
    except Exception as e:
        print(f"Error: {e}")


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


def get_encoded_image(image_path):
    img = cv2.imread(image_path)
    # Encode into PNG and send to ControlNet
    retval, bytes = cv2.imencode('.png', img)
    return base64.b64encode(bytes).decode('utf-8')


def run_preprocessor(preprocessor_name, image_path, current_user_id, filename="preprocessed.jpg", res=512):
    input_image = get_encoded_image(image_path)
    data = {
        "controlnet_module": preprocessor_name,
        "controlnet_input_images": [input_image],
        "controlnet_processor_res": res,
        "controlnet_threshold_a": 64,
        "controlnet_threshold_b": 64
    }
    preprocessor_url = 'http://127.0.0.1:7861/controlnet/detect'
    response = submit_post(preprocessor_url, data)
    output_dir = f"disruptor/static/images/{current_user_id}/preprocessed"
    output_filepath = os.path.join(output_dir, filename)

    # If there was no such dir, we create it and try again
    try:
        save_encoded_image(response.json()['images'][0], output_filepath)
    except FileNotFoundError as e:
        create_directory_if_not_exists(output_dir)
        save_encoded_image(response.json()['images'][0], output_filepath)


def convert_to_mask(image_path, output_path=None):
    # Open the image
    image = Image.open(image_path)

    # Convert the image to grayscale
    grayscale_image = image.convert("L")

    # Threshold the grayscale image
    threshold = 128  # You can adjust this threshold value as needed
    mask = grayscale_image.point(lambda p: 0 if p > threshold else 255)

    # Save the mask to the same path as the original image
    if output_path is None:
        mask.save(image_path)
    else:
        mask.save(output_path)


def find_closest_point(contour, reference_point):
    min_distance = float('inf')
    closest_point = None
    for point in contour:
        distance = np.linalg.norm(point[0] - reference_point)
        if distance < min_distance:
            min_distance = distance
            closest_point = point[0]
    return closest_point


def find_matching_points(contour, p1, p2, tolerance=5, distance_threshold=20):
    # We know that our contours are just quadrilaterals, so for one x there are 2 possible points with different y.
    # (actually more if it is right or left edge and it is vertical, but we put our furniture bounding box projection
    # in the middle always and it is smaller than wall_mask so it is 2 points only possible)
    # But contour border has small gaps, so we add tolerance for x axis
    # Since we add tolerance for x, sometimes it finds points that have similar x and also similar y coordinates
    # So we filter the points and leave only those which are further away from each other by using distance_threshold

    contour = np.squeeze(contour)  # Ensure 2D array

    # Find indices where x-coordinate is within the tolerance for point1 and point2
    indices_p1 = np.where(np.abs(contour[:, 0] - p1[0]) <= tolerance)[0]
    indices_p2 = np.where(np.abs(contour[:, 0] - p2[0]) <= tolerance)[0]

    matching_p1 = []
    matching_p1.extend(contour[indices_p1])

    matching_p2 = []
    matching_p2.extend(contour[indices_p2])

    unique_p1 = []
    for point in matching_p1:
        if not any(np.linalg.norm(point - u_point) < distance_threshold for u_point in unique_p1):
            unique_p1.append(point)

    unique_p2 = []
    for point in matching_p2:
        if not any(np.linalg.norm(point - u_point) < distance_threshold for u_point in unique_p2):
            unique_p2.append(point)

    return np.array(unique_p1), np.array(unique_p2)


def get_corners(contour, wall_mask):
    empty_mask = np.zeros_like(wall_mask)
    # cv2.drawContours(mask_approx, [approx], -1, 255, -1)
    # Convert single-channel mask to 3-channel for Harris corner detection
    cv2.drawContours(empty_mask, [contour], -1, (0, 255, 0), 3)
    # cv2.imshow('123', mask_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    mask_gray = cv2.cvtColor(empty_mask, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(mask_gray, 10, 9, 0.04)
    dst = cv2.dilate(dst, None)
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(mask_bgr, contours, 0, (0, 255, 0), 3)
    # Find coordinates of corners
    corner_pixels = np.argwhere(dst > 0.01 * dst.max())

    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=25, min_samples=3).fit(corner_pixels)
    cluster_labels = clustering.labels_

    # Dictionary to store clustered points
    clustered_points = {}

    # Collect points into clusters
    for i, cluster_number in enumerate(cluster_labels):
        if cluster_number != -1:  # Ignore noise points
            if cluster_number not in clustered_points:
                clustered_points[cluster_number] = []
            clustered_points[cluster_number].append(corner_pixels[i])

    # Calculate cluster centers (average position)
    cluster_centers = dict()
    for cluster_number in clustered_points.keys():
        cluster = np.array(clustered_points[cluster_number])
        center = np.mean(cluster, axis=0, dtype=np.int32)
        cluster_centers[cluster_number] = np.flip(center)  # To make it x,y and not y,x

    return [x_y for x_y in cluster_centers.values()]


def find_lowest_point(points):
    # Convert the points array to a NumPy array for easier manipulation
    points = np.array(points)

    # Use np.argmin() to get the index of the point with the highest y-coordinate
    index_of_lowest = np.argmax(
        points[:, 1])  # We use argmax because, the up left corner is 0,0 and the bigger the y is the lower the point is

    # Use the index to get the lowest point
    lowest_point = points[index_of_lowest]

    return lowest_point


# def find_bed_placement_coordinates(bed_back_mask_path, wall_mask_path, save_path=None):
#     # Load images
#     bed_back_mask = cv2.imread(bed_back_mask_path, cv2.IMREAD_GRAYSCALE)
#     wall_mask = cv2.imread(wall_mask_path, cv2.IMREAD_GRAYSCALE)
#
#     # Find contours in both masks
#     bed_back_contours, _ = cv2.findContours(bed_back_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     wall_contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Calculate centroid of bed contour
#     bed_back_centroid = np.mean(bed_back_contours[0], axis=0)[0]
#
#     # Calculate offset from bed centroid to wall centroid
#     wall_centroid = np.mean(wall_contours[0], axis=0)[0]
#     offset_x = wall_centroid[0] - bed_back_centroid[0]
#
#     # Find lowest possible y-coordinate for the bed within wall boundaries
#     bed_contour = bed_back_contours[0]
#     wall_contour = wall_contours[0]
#
#     # Shift bed contour along x-axis
#     bed_contour_shifted = bed_contour.copy()
#     bed_contour_shifted[:, 0, 0] += int(offset_x)
#
#     wall_mask = cv2.imread(wall_mask_path)
#     ordered_corners = order_points(np.array(get_corners(bed_contour_shifted, wall_mask)))
#     br = ordered_corners[2]
#     bl = ordered_corners[3]
#     print(br, bl)
#
#     matching_br, matching_bl = find_matching_points(wall_contour, br, bl, 2)
#     print(matching_br, matching_bl)
#
#     lowest_for_br = find_lowest_point(matching_br)
#     lowest_for_bl = find_lowest_point(matching_bl)
#     print(lowest_for_br, lowest_for_bl)
#     offset_br_y = int(lowest_for_br[1] - br[1])
#     offset_bl_y = int(lowest_for_bl[1] - bl[1])
#     print(offset_br_y, offset_bl_y)
#     offset_y = max(offset_br_y, offset_bl_y)
#     print(offset_y)
#
#     if save_path:
#         contours_image = cv2.imread(wall_mask_path)
#         cv2.drawContours(contours_image, [wall_contour, bed_contour], -1, (255, 0, 0), 5)
#         cv2.imwrite(save_path + "/contours_before.png", contours_image)
#
#         contours_image = cv2.imread(wall_mask_path)
#         cv2.drawContours(contours_image, [wall_contour, bed_contour_shifted], -1, (255, 0, 0), 5)
#         cv2.imwrite(save_path + "/contours_x.png", contours_image)
#
#         bed_contour_xy_shifted = bed_contour_shifted.copy()
#         bed_contour_xy_shifted[:, 0, 1] += int(offset_y)
#         contours_image = cv2.imread(wall_mask_path)
#         cv2.drawContours(contours_image, [wall_contour, bed_contour_xy_shifted], -1, (255, 0, 0), 5)
#         cv2.imwrite(save_path + "/contours_xy.png", contours_image)
#     return int(offset_x), int(offset_y)

# def find_bed_placement_coordinates(wall_mask_path):
#     wall_mask = cv2.imread(wall_mask_path, cv2.IMREAD_GRAYSCALE)
#     wall_contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Calculate offset from bed centroid to wall centroid
#     wall_centroid = np.mean(wall_contours[0], axis=0)[0]
#     pixel_x = wall_centroid[0]
#     pixel_y = wall_centroid[1]
#
#     return int(pixel_x), int(pixel_y)


def create_mask_of_size(width, height):
    # Create a new black image with the same size
    black_mask = Image.new("RGB", (width, height), color=(0, 0, 0))
    return black_mask


def create_furniture_mask(es_path, furniture_renders_paths: list, furniture_renders_offsets: list, save_path):
    try:
        image = Image.open(es_path)
        width, height = image.size
        empty_mask = create_mask_of_size(width, height)
        print("Saving empty mask to:", save_path)
        os.remove(save_path)
        empty_mask.save(save_path)
        print("Empty mask saved successfully!")
    except Exception as e:
        print("An error occurred while saving the mask:", e)
    for i in range(len(furniture_renders_paths)):
        mask_path = os.path.join(os.path.dirname(save_path), 'furniture_mask.png')
        convert_to_mask(furniture_renders_paths[i], mask_path)
        overlay_masks(mask_path, save_path, save_path, furniture_renders_offsets[i])
    from disruptor.preprocess_for_empty_space import perform_dilation
    perform_dilation(save_path, save_path, 32)


def overlay_masks(fg_mask_path, bg_mask_path, output_path, coordinates):
    # Open both images
    img1 = Image.open(fg_mask_path)
    img2 = Image.open(bg_mask_path)

    # Remove white background from the first image
    img1 = img1.convert("RGBA")
    datas = img1.getdata()
    newData = []
    for item in datas:
        # Change all white (also shades of whites)
        # pixels to transparent
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img1.putdata(newData)

    # Paste img1 onto img2 at the specified coordinates
    img2.paste(img1, coordinates, mask=img1)

    # Save or display the resulting image
    # img2.show()  # Display the resulting image
    img2.save(output_path)  # Save the resulting image


def overlay_images(fg_image_path, bg_image_path, output_path, coordinates):
    # Open both images
    img1 = Image.open(fg_image_path)
    img2 = Image.open(bg_image_path)

    # Remove white background from the first image
    img1 = img1.convert("RGBA")
    datas = img1.getdata()
    newData = []
    for item in datas:
        # Change all white (also shades of whites)
        # pixels to transparent
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img1.putdata(newData)

    # Paste img1 onto img2 at the specified coordinates
    img2.paste(img1, coordinates, mask=img1)

    # Save or display the resulting image
    # img2.show()  # Display the resulting image
    img2.save(output_path)  # Save the resulting image


def get_image_size(image_path):
    image = Image.open(image_path)
    width, height = image.size
    image.close()
    return width, height

def find_abs_min_z(points):
    # Find the minimum z-coordinate among the points
    min_z = np.min(points[:, 2])
    return abs(min_z)

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


def image_overlay(furniture_image, background_image):
    # Assume that both images are in PNG format
    # Resize the fist_image to match the size of the second_image
    furniture_image = furniture_image.resize(background_image.size)

    # Overlay the decoded image on top of the background image
    combined_image = Image.alpha_composite(background_image.convert('RGBA'), furniture_image.convert('RGBA'))

    return combined_image