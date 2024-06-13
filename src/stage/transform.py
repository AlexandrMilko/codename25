# import the necessary packages
import numpy as np
import cv2
from tools import order_points, get_image_size

def plot_points(pts):
    import matplotlib.pyplot as plt
    colors = ['red', 'green', 'blue', 'black']

    plt.figure()
    for i, pt in enumerate(pts):
        plt.scatter(pt[0], pt[1], color=colors[i], label=f'{colors[i]} point')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()


def extract_angle(H):
    # print(H)
    angle_rad = np.arctan2(H[1, 0], H[0, 0])
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def abs_sum(lst):
    return sum(abs(x) for x in lst)


def yaw_rot(lst):
    return abs(lst[1])


def pitch_rot(lst):
    return abs(lst[0])

def extract_angle_new(H, height, width):
    import stage.shapeUtil as su
    from stage.depth_estimation import get_intrinsics
    # Define the camera matrix (example)
    # K = np.matrix([
    #     [476.7, 0.0, 400.0],
    #     [0.0, 476.7, 400.0],
    #     [0.0, 0.0, 1.0]])
    K = get_intrinsics(height, width)

    (R, T) = su.decHomography(K, H)
    Rot = su.decRotation(R)
    return -Rot[0] * 180 / np.pi, -Rot[1] * 180 / np.pi, -Rot[2] * 180 / np.pi

# def extract_angle_new(H):
#     # Define the camera matrix (example)
#     K = np.array([[4048, 0, 2587],
#                   [0, 4046, 1556],
#                   [0, 0, 1]])
#
#     # Decompose the homography matrix
#     _, rot_matrices, _, _ = cv2.decomposeHomographyMat(H, K)
#     # Extract rotation angles from all possible rotation matrices
#     possible_rotations = []
#     for rot_matrix in rot_matrices:
#         # angle_rad = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
#         # angle_deg = np.degrees(angle_rad)
#         # angles_deg.append(angle_deg)
#         rvec, _ = cv2.Rodrigues(rot_matrix)
#         xyz_degrees = np.degrees(rvec.flatten())
#         possible_rotations.append(xyz_degrees)
#
#     print(possible_rotations)
#
#     return min(possible_rotations, key=pitch_rot)


# def four_point_transform(image, pts):
def four_point_transform(pts, image_path):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)

    # print(tl, tr, br, bl)
    from PIL import Image
    width, height = get_image_size(image_path)
    xyz_deg = extract_angle_new(M, height, width)
    # print(xyz_deg, "ANGLES")

    # warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    # return warped, xyz_deg
    return xyz_deg
