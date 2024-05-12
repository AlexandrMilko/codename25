import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import gc

def pixel_to_3d(camera_intrinsics, depth_image, pixel_coord):
    """Converts a 2D pixel coordinate to 3D position relative to camera center.

    Args:
        camera_intrinsics: 3x3 numpy array, camera intrinsic matrix.
        depth_image: 2D numpy array, depth map.
        pixel_coord: Tuple (x, y) of pixel coordinates.

    Returns:
        3D position as a numpy array [X, Y, Z] relative to camera center.
    """
    x, y = pixel_coord
    depth = depth_image[y, x]  # Extract depth at pixel (x, y)

    # Create homogeneous pixel coordinate [x, y, 1]
    pixel_homogeneous = np.array([x, y, 1], dtype=np.float32)

    # Convert pixel coordinate to normalized device coordinates (NDC)
    pixel_ndc = np.linalg.inv(camera_intrinsics) @ pixel_homogeneous

    # Convert NDC to 3D point in camera space (with depth)
    camera_point = np.array([pixel_ndc[0], pixel_ndc[1], 1.0]) * depth
    # z_offset = 2.5
    # camera_point[2] -= z_offset
    # x - points to the right
    # y - bigger it is the lower is our point compared to camera y
    # z - shows the distance from camera
    return camera_point


def get_pixel_3d_coords(camera_intrinsics, depth_image):
    """
    Get 3D coordinates for each pixel in the image.

    Args:
        camera_intrinsics: Camera intrinsic matrix.
        depth_image: Depth map.

    Returns:
        List of 3D coordinates for each pixel.
    """
    h, w = depth_image.shape

    # Create arrays to store 3D coordinates
    pixel_coords_3d = []

    for y in range(h):
        for x in range(w):
            # Calculate 3D coordinates for each pixel
            pixel_coord = (x, y)
            pixel_3d = transform_to_blender_xyz(*pixel_to_3d(camera_intrinsics, depth_image, pixel_coord))
            pixel_coords_3d.append(pixel_3d)

    return pixel_coords_3d

def plot_pixels_3d(points_filepath, target_point=None):

    # Load 3D points from the text file
    points = np.loadtxt(points_filepath, delimiter=',')

    selected_points = points[::50]

    # Plot 3D points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(selected_points[:, 0], selected_points[:, 1], selected_points[:, 2], alpha=0.1)
    if target_point is not None:
        ax.scatter(*target_point, alpha=1, color='red', s=100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def transform_to_blender_xyz(x, y, z):
    # 1. Invert the y
    # 2. Swap the z and y
    return x, z, -y

def predict_depth(model, image):
    depth = model.infer_pil(image)
    return depth

def depth_to_points(depth, R=None, t=None):

    K = get_intrinsics(depth.shape[1], depth.shape[2])
    Kinv = np.linalg.inv(K)
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)

    # M converts from your coordinate to PyTorch3D's coordinate system
    M = np.eye(3)
    M[0, 0] = -1.0
    M[1, 1] = -1.0

    height, width = depth.shape[1:3]

    x = np.arange(width)
    y = np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
    coord = coord.astype(np.float32)
    # coord = torch.as_tensor(coord, dtype=torch.float32, device=device)
    coord = coord[None]  # bs, h, w, 3

    D = depth[:, :, :, None, None]
    # print(D.shape, Kinv[None, None, None, ...].shape, coord[:, :, :, :, None].shape )
    pts3D_1 = D * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]
    # pts3D_1 live in your coordinate system. Convert them to Py3D's
    pts3D_1 = M[None, None, None, ...] @ pts3D_1
    # from reference to targe tviewpoint
    pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :, None]
    # pts3D_2 = pts3D_1
    # depth_2 = pts3D_2[:, :, :, 2, :]  # b,1,h,w
    return pts3D_2[:, :, :, :3, 0][0]

def create_triangles(h, w, mask=None):
    """Creates mesh triangle indices from a given pixel grid size.
        This function is not and need not be differentiable as triangle indices are
        fixed.
    Args:
    h: (int) denoting the height of the image.
    w: (int) denoting the width of the image.
    Returns:
    triangles: 2D numpy array of indices (int) with shape (2(W-1)(H-1) x 3)
    """
    x, y = np.meshgrid(range(w - 1), range(h - 1))
    tl = y * w + x
    tr = y * w + x + 1
    bl = (y + 1) * w + x
    br = (y + 1) * w + x + 1
    triangles = np.array([tl, bl, tr, br, tr, bl])
    triangles = np.transpose(triangles, (1, 2, 0)).reshape(
        ((w - 1) * (h - 1) * 2, 3))
    if mask is not None:
        mask = mask.reshape(-1)
        triangles = triangles[mask[triangles].all(1)]
    return triangles

def get_intrinsics(H,W):
    # TODO change fov to typical cell phones fov
    """
    Intrinsics for a pinhole camera model.
    Assume fov of 55 degrees and central principal point.
    """
    f = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]])

def image_pixel_list_to_3d(image_path, pixels_coordinates: list[list[int,int]]):
    from PIL import Image
    with torch.no_grad():
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torch.hub.load('isl-org/ZoeDepth', "ZoeD_NK", pretrained=True).to(DEVICE).eval()
        image = Image.open(image_path)

        # TODO play with this thumbnail if your depth 3d representation is squeezed
        image.thumbnail((1024, 1024))  # limit the size of the input image
        depth = predict_depth(model, image)
        print("DEPTH PREDICTED FOR WALL CORNERS")
        camera_intrinsics = get_intrinsics(depth.shape[0], depth.shape[1])
        points_3d = []
        for x, y in pixels_coordinates:
            print("POINT being processed: ", x, y)
            point_3d = transform_to_blender_xyz(*pixel_to_3d(camera_intrinsics, depth, (x, y)))
            print(f"{x, y} -> {point_3d}")
            points_3d.append(point_3d)
        print("ALL POINTS WERE ADDED")
        del depth
        del image
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print("RETURNED")
        return points_3d

def image_pixel_to_3d(image_path, pixel_coordinates):
    from PIL import Image
    with torch.no_grad():
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torch.hub.load('isl-org/ZoeDepth', "ZoeD_NK", pretrained=True).to(DEVICE).eval()
        image = Image.open(image_path)

        # TODO play with this thumbnail if your depth 3d representation is squeezed
        image.thumbnail((1024, 1024))  # limit the size of the input image
        depth = predict_depth(model, image)

        camera_intrinsics = get_intrinsics(depth.shape[0], depth.shape[1])
        pixel_3d = transform_to_blender_xyz(*pixel_to_3d(camera_intrinsics, depth, pixel_coordinates))

        del depth
        del image
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return pixel_3d
def image_pixels_to_3d(image_path, output_fname):
    from PIL import Image
    with torch.no_grad():
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torch.hub.load('isl-org/ZoeDepth', "ZoeD_NK", pretrained=True).to(DEVICE).eval()
        image = Image.open(image_path)

        # TODO play with this thumbnail if your depth 3d representation is squeezed
        image.thumbnail((1024, 1024))  # limit the size of the input image
        depth = predict_depth(model, image)

        camera_intrinsics = get_intrinsics(depth.shape[0], depth.shape[1])
        print(depth.shape[0], depth.shape[1], "depth IMAGE size")

        pixel_coords_3d = get_pixel_3d_coords(camera_intrinsics, depth)
        with open(output_fname, "w") as f:
            for coord in pixel_coords_3d:
                f.write(f"{coord[0]},{coord[1]},{coord[2]}\n")

        del depth
        del image
        del model
        gc.collect()
        torch.cuda.empty_cache()

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

if __name__ == "__main__":
    image_pixels_to_3d('10_empty.jpg', '3d_coords.txt')
    rotate_3d_points('3d_coords.txt', '3d_coords_rotated.txt', np.deg2rad(-18.33465), np.deg2rad(-15))
