import cv2
import numpy as np


class Floor:
    def __init__(self, centroid: list[int], seg_image_path: str):
        self.centroid = centroid
        self.seg_image_path = seg_image_path

    @staticmethod
    def save_mask(seg_image_path, save_path):
        # Load the image
        image = cv2.imread(seg_image_path)

        lower_bound = np.array([45, 45, 75])
        upper_bound = np.array([55, 55, 85])

        # Create a binary mask using inRange
        bw_mask = cv2.inRange(image, lower_bound, upper_bound)

        # Apply additional actions to the mask
        blurred = cv2.GaussianBlur(bw_mask, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(thresh, kernel, iterations=1)
        final_mask = cv2.dilate(erosion, kernel, iterations=1)

        # Save the binary mask
        cv2.imwrite(save_path, final_mask)

    @staticmethod
    def find_centroid(seg_image_path):
        save_path = 'disruptor/images/floor_mask.png'
        Floor.save_mask(seg_image_path, save_path)
        image = cv2.imread(save_path, cv2.IMREAD_GRAYSCALE)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contour = contours[0]
            # Calculate the moments of the contour
            M = cv2.moments(contour)

            # Calculate centroid
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroid = (cx, cy)
                return centroid
            else:
                raise Exception("The contour has no area, centroid cannot be calculated.")
        else:
            raise Exception("No contours found in the image.")