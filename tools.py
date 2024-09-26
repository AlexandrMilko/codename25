from PIL import Image
import open3d as o3d
import numpy as np
import requests
import base64
import json
import cv2
import os


def calculate_pitch_angle(plane_normal):
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    pitch_angle_rad = np.arctan2(plane_normal[1], plane_normal[2])
    pitch_angle_deg = np.degrees(pitch_angle_rad)

    return pitch_angle_deg


def calculate_roll_angle(plane_normal, reference_vector=[1, 0, 0]):
    # Step 1: Normalize the normal vector of the plane and the reference vector
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    reference_vector = reference_vector / np.linalg.norm(reference_vector)

    # Step 2: Project the plane normal onto the XZ plane (remove Y component)
    plane_normal_proj_xz = np.array([plane_normal[0], 0, plane_normal[2]])

    # Step 3: Normalize the projected vector
    plane_normal_proj_xz = plane_normal_proj_xz / np.linalg.norm(plane_normal_proj_xz)

    # Step 4: Calculate the dot product between the reference vector and the projected normal
    dot_product = np.dot(reference_vector, plane_normal_proj_xz)

    # Step 5: Compute the roll angle using the arccosine of the dot product
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
    inlier_cloud = point_cloud.select_by_index(inliers)
    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)

    inlier_cloud.paint_uniform_color([1, 0, 0])  # Paint plane points in red
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


def run_preprocessor(preprocessor_name, input_path, output_filepath, SD_DOMAIN, res=512):
    input_image = get_encoded_image(input_path)
    data = {
        "controlnet_module": preprocessor_name,
        "controlnet_input_images": [input_image],
        "controlnet_processor_res": res,
        "controlnet_threshold_a": 64,
        "controlnet_threshold_b": 64
    }
    preprocessor_url = f'http://{SD_DOMAIN}:7861/controlnet/detect'
    response = submit_post(preprocessor_url, data)

    save_encoded_image(response.json()['images'][0], output_filepath)


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


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


def get_encoded_image_from_path(image_path):
    img = cv2.imread(image_path)
    # Encode into PNG and send to ControlNet
    try:
        retval, bytes = cv2.imencode('.png', img)
    except cv2.error:
        retval, bytes = cv2.imencode('.jpg', img)
    return base64.b64encode(bytes).decode('utf-8')


def convert_png_to_mask(image_path, output_path=None):
    # Open the image
    image = Image.open(image_path).convert("RGBA")

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Extract the alpha channel
    alpha_channel = image_array[:, :, 3]

    # Create a mask based on the alpha channel
    mask_array = np.where(alpha_channel > 0, 255, 0).astype(np.uint8)

    # Convert the mask array back to an image
    mask = Image.fromarray(mask_array, mode='L')

    # Save the mask to the specified path or overwrite the original image
    if output_path is None:
        output_path = image_path
    mask.save(output_path)


def save_mask_of_size(width, height, output_path):
    # Create a new black image with the same size
    black_mask = Image.new("RGB", (width, height), color=(0, 0, 0))
    print("Saving empty mask to:", output_path)
    black_mask.save(output_path)
    print("Empty mask saved successfully!")
    return black_mask


def overlay_masks(fg_mask_path, bg_mask_path, output_path):
    # Open both images
    fg_img = Image.open(fg_mask_path).convert("RGBA")
    bg_img = Image.open(bg_mask_path).convert("RGBA")

    # Convert images to numpy arrays
    fg_img_array = np.array(fg_img)
    bg_img_array = np.array(bg_img)

    # Create a mask for black pixels in the foreground image
    black_mask = (fg_img_array[:, :, :3] == 0).all(axis=2)

    # Set black pixels to transparent
    fg_img_array[black_mask] = [255, 255, 255, 0]

    # Determine the dimensions of the overlay area
    fg_h, fg_w = fg_img_array.shape[:2]
    bg_h, bg_w = bg_img_array.shape[:2]

    # Ensure the overlay does not exceed background dimensions
    overlay_h = min(fg_h, bg_h)
    overlay_w = min(fg_w, bg_w)

    # Overlay the images
    bg_img_array[:overlay_h, :overlay_w] = np.where(
        fg_img_array[:overlay_h, :overlay_w, 3:] > 0,
        fg_img_array[:overlay_h, :overlay_w],
        bg_img_array[:overlay_h, :overlay_w]
    )

    # Convert the result back to an image
    result_img = Image.fromarray(bg_img_array, 'RGBA')

    # Save the resulting image
    result_img.save(output_path)


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


def overlay_image(furniture_image, background_image):
    # Assume that both images are in PNG format
    # Resize the fist_image to match the size of the second_image
    furniture_image = furniture_image.resize(background_image.size)

    # Overlay the decoded image on top of the background image
    combined_image = Image.alpha_composite(background_image.convert('RGBA'), furniture_image.convert('RGBA'))

    return combined_image


def restart_stable_diffusion(api_url: str):
    import time
    """
    Restart the Stable Diffusion WebUI using its API.
    """
    restart_endpoint = f"{api_url}/sdapi/v1/restart"

    response = requests.post(restart_endpoint)

    if response.status_code == 200:
        print("Restart command sent successfully.")
    else:
        print(f"Failed to send restart command. Status code: {response.status_code}")

    # Wait for the server to restart
    time.sleep(10)

    # Verify if the server is up
    try:
        health_check_url = f"{api_url}/healthcheck"
        health_response = requests.get(health_check_url)
        if health_response.status_code == 200:
            print("Server restarted successfully.")
        else:
            print("Server is still down. Please check manually.")
    except requests.exceptions.RequestException as e:
        print(f"Exception during health check: {e}")
