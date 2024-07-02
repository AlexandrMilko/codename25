import cv2
import numpy as np
import matplotlib.pyplot as plt
from tools import calculate_angle_from_top_view

class Wall:
    def __init__(self, corners: list[list[int]], image_path: str, centroids: list[list[int]]):
        self.corners = corners
        self.image_path = image_path
        self.left_centroid, self.right_centroid = centroids

    def find_angle_from_3d(self, room, pitch_rad, roll_rad):
        # # We consider only top corners. Taking bottom corners additionally doesnt give us any more useful info
        # top_left, top_right = self.corners[:2]

        # We use left and right centroids,
        # because sometimes depth estimation models is not accurate and it thinks
        # corner is a pixel on a wall next to our wall, so it outputs wrong 3D position
        # Additionally, sometimes our corners are estimated incorrectly due to watershed algo errors when
        # there is a small wall next to our wall
        try:
            from DepthAnything.depth_estimation import image_pixel_list_to_3d, rotate_3d_point
            points_3d = image_pixel_list_to_3d(room.empty_room_image_path, [self.left_centroid, self.right_centroid])
            points_3d_before_camera_rotation = [rotate_3d_point(point, -pitch_rad, -roll_rad) for point in points_3d]
        except Exception as e:
            raise e

        points_3d_before_camera_rotation = np.array(points_3d_before_camera_rotation)
        return calculate_angle_from_top_view(*points_3d_before_camera_rotation)

    def save_mask(self, save_path):
        from PIL import Image, ImageDraw
        # Open the image
        image = Image.open(self.image_path)

        # Create a new blank image with the same size as the original image
        mask = Image.new("RGB", image.size, color=(0, 0, 0))

        # Create a draw object
        draw = ImageDraw.Draw(mask)
        corners = [tuple(point) for point in self.corners]
        # Draw a quadrilateral
        draw.polygon(corners, fill=(255, 255, 255))

        # Save the masked image
        mask.save(save_path)

    @staticmethod
    def find_walls(seg_img_path):
        # We separate the walls from each other from a segmented image
        from scipy import ndimage
        from skimage.feature import peak_local_max
        # import matplotlib.pyplot as plt
        from skimage.segmentation import watershed

        # Read the image
        image = cv2.imread(seg_img_path)

        # Define the color range to filter
        lower_color = np.array([110, 110, 110])  # Wall colors
        upper_color = np.array([130, 130, 130])

        # Mask the image to extract pixels within the specified color range
        image = cv2.inRange(image, lower_color, upper_color)

        distance = ndimage.distance_transform_edt(image)
        coords = peak_local_max(distance, footprint=np.ones((10, 10)), labels=image)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndimage.label(mask)
        labels = watershed(-distance, markers, mask=image)

        # fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
        # ax = axes.ravel()

        # ax[0].imshow(image, cmap=plt.cm.gray)
        # ax[0].set_title('Overlapping objects')
        # ax[1].imshow(-distance, cmap=plt.cm.gray)
        # ax[1].set_title('Distances')
        # ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
        # ax[2].set_title('Separated objects')

        # for a in ax:
        #     a.set_axis_off()

        filtered_classes = Wall.filter_walls(labels)

        walls_corners, walls_centroids = Wall.find_corners_of_labeled_regions(labels, filtered_classes)

        walls = []
        for label in walls_corners.keys():
            wall_corners = walls_corners[label]
            wall_centroids = walls_centroids[label]
            walls.append(Wall(wall_corners, seg_img_path, wall_centroids))

        # fig.tight_layout()
        # plt.show()
        return walls

    def __repr__(self):
        return f"Wall({self.image_path}): {self.corners}"

    @staticmethod
    def filter_walls(label_array, min_area_proportion=0.1):
        pixel_number = np.prod(label_array.shape, axis=0)
        min_pixel_number = pixel_number * min_area_proportion
        unique_classes, counts = np.unique(label_array, return_counts=True)
        filtered_classes = unique_classes[counts >= min_pixel_number]
        return filtered_classes

    @staticmethod
    def get_biggest_wall_class(label_array):
        unique_classes, counts = np.unique(label_array, return_counts=True)
        # Filter out the class 0
        mask = unique_classes != 0
        filtered_classes = unique_classes[mask]
        filtered_counts = counts[mask]

        # Find the class with the highest count among the filtered classes
        if filtered_counts.size == 0:
            return None  # Or handle the case where there are no valid classes

        max_count_index = np.argmax(filtered_counts)
        class_with_max_count = filtered_classes[max_count_index]

        return class_with_max_count

    @staticmethod
    def find_biggest_wall(seg_img_path):
        # We separate the walls from each other from a segmented image
        from scipy import ndimage
        from skimage.feature import peak_local_max
        # import matplotlib.pyplot as plt
        from skimage.segmentation import watershed

        # Read the image
        image = cv2.imread(seg_img_path)

        # Define the color range to filter
        lower_color = np.array([110, 110, 110])  # Wall colors
        upper_color = np.array([130, 130, 130])

        # Mask the image to extract pixels within the specified color range
        image = cv2.inRange(image, lower_color, upper_color)

        distance = ndimage.distance_transform_edt(image)
        coords = peak_local_max(distance, footprint=np.ones((10, 10)), labels=image)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndimage.label(mask)
        labels = watershed(-distance, markers, mask=image)

        biggest_wall_class = Wall.get_biggest_wall_class(labels)
        print(labels, "labels")
        print(biggest_wall_class, "biggest_wall_class")

        walls_corners, walls_centroids = Wall.find_corners_of_labeled_regions(labels, [biggest_wall_class])

        return Wall(walls_corners[biggest_wall_class], seg_img_path, walls_centroids[biggest_wall_class])

    @staticmethod
    def find_corners_of_labeled_regions(label_array, filtered_classes):
        walls_corners = dict()
        walls_centroids = dict()
        for label in filtered_classes:
            if label == 0: continue
            mask = np.uint8(label_array == label) * 255  # Convert to 0 or 255

            # Approximate contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            epsilon = 0.05 * cv2.arcLength(contours[0], True)  # Adjust epsilon as needed
            approx = cv2.approxPolyDP(contours[0], epsilon, True)
            centroids = Wall.find_left_right_centroids(contours[0])
            mask_approx = np.zeros_like(mask)
            # cv2.drawContours(mask_approx, [approx], -1, 255, -1)
            # Convert single-channel mask to 3-channel for Harris corner detection
            mask_bgr = cv2.merge([mask_approx, mask_approx, mask_approx])
            cv2.drawContours(mask_bgr, [approx], -1, (0, 255, 0), 3)
            # cv2.imshow('123', mask_bgr)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            mask_gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
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
                cluster_centers[cluster_number] = np.flip(center) # To make it x,y and not y,x

            walls_corners[label] = [x_y for x_y in cluster_centers.values()]
            walls_centroids[label] = centroids

        return walls_corners, walls_centroids
    @staticmethod
    def find_left_right_centroids(wall_contour):
        M = cv2.moments(wall_contour)
        centroid_x = int(M["m10"] / M["m00"])

        contour_left_half = wall_contour[wall_contour[:, :, 0] < centroid_x]
        contour_right_half = wall_contour[wall_contour[:, :, 0] >= centroid_x]

        # Compute moments for left and right halves
        moments_left = cv2.moments(contour_left_half)
        moments_right = cv2.moments(contour_right_half)

        # Calculate centroids
        centroid_left = (
            int(moments_left["m10"] / moments_left["m00"]), int(moments_left["m01"] / moments_left["m00"]))
        centroid_right = (
            int(moments_right["m10"] / moments_right["m00"]), int(moments_right["m01"] / moments_right["m00"]))

        return centroid_left, centroid_right

if __name__ == "__main__":
    walls = Wall.find_walls('../test_imgs/39Before.jpg')
    # walls = Wall.find_walls('../test_imgs/88Before.jpg')
    # walls = Wall.find_walls('../test_imgs/26Before.jpg')
    # walls = Wall.find_walls('test_imgs/129Before.jpg')
    # walls = Wall.find_walls('test_imgs/349Before.jpg')

    for wall in walls:
        print(wall.find_angle_from_2d())