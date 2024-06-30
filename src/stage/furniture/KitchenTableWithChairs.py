from stage.furniture.Furniture import FloorFurniture
import cv2
import numpy as np

class KitchenTableWithChairs(FloorFurniture):
    # We use it to scale the model to metric units
    scale = 1, 1, 1
    # We use it to compensate the angle if the model is originally rotated in a wrong way
    default_angles = 0, 0, 0

    def __init__(self, model_path='3Ds/kitchen/kitchen_table_with_chairs.usdc'):
        super().__init__(model_path)

    @staticmethod
    def find_centers(segments):
        centers = []
        for segment in segments:
            M = cv2.moments(segment)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append([cX, cY])

        return centers

    @staticmethod
    def choose_bigger_segments(segments, total_area):
        segments.sort(key=lambda x: x[1], reverse=True)

        chosen_segments = []
        cumulative_area = 0
        # Change value in order to change the number of places where to put the table
        max_area = 0.9 * total_area
        for segment, area in segments:
            if cumulative_area + area <= max_area:
                chosen_segments.append(segment)
                cumulative_area += area
            else:
                break

        return chosen_segments

    @staticmethod
    def find_segments(contours, bottom_point):
        segments = []
        for i in range(len(contours)):
            start_point = tuple(contours[i][0])
            end_point = tuple(contours[(i + 1) % len(contours)][0])
            # (i + 1) % len(approx) Ensuring it wraps around to
            # the first vertex when processing the last vertex

            segment = np.array([
                start_point,  # Left upper point
                end_point,  # Right upper point
                (end_point[0], bottom_point),  # Right bottom point
                (start_point[0], bottom_point)  # Left bottom point
            ], np.int32)

            area = cv2.contourArea(segment)
            segments.append((segment, area))

        return segments

    @staticmethod
    def find_placement_pixel(floor_mask_path: str) -> list[list[int, int]]:
        image = cv2.imread(floor_mask_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]

        total_area = cv2.contourArea(contour)

        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        segments = KitchenTableWithChairs.find_segments(approx, image.shape[0])
        chosen_segments = KitchenTableWithChairs.choose_bigger_segments(segments, total_area)
        centers = KitchenTableWithChairs.find_centers(chosen_segments)

        # # Draw the segments and centers on the original image
        # for segment in chosen_segments:
        #     cv2.polylines(image, [segment], isClosed=True, color=(0, 255, 0), thickness=2)
        # for center in centers:
        #     cv2.circle(image, center, 5, (0, 0, 255), -1)
        #
        # cv2.imshow('Segments and Centers', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return centers
    @staticmethod
    def find_placement_pixel_from_floor_layout(window_mask_path: str) -> list[list[tuple[int, int]]]:
        image = cv2.imread(window_mask_path, cv2.IMREAD_GRAYSCALE)

        origin = (image.shape[1] // 2, image.shape[0] // 2)
        angle = KitchenTableWithChairs.find_angle(image)

        x_coords, y_coords, square_size = KitchenTableWithChairs.crate_grid(image)
        rotated_coords = KitchenTableWithChairs.rotate_coordinates(x_coords, y_coords, angle, origin)
        squares = KitchenTableWithChairs.find_squares(rotated_coords, square_size, image)
        centers = KitchenTableWithChairs.find_square_center(squares)

        # Draw & Display
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for square in squares:
            x, y, size = square
            cv2.rectangle(color_image, (x, y), (x + size, y + size), (0, 255, 0), 1)

        for center in centers:
            cv2.circle(color_image, center, 3, (0, 0, 255), -1)

        cv2.imwrite('images/preprocessed/floor_layout_debug.png', color_image)

        return centers
    @staticmethod
    def find_angle(image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Calculate the sides of the rectangle
        sides = [(box[i], box[(i + 1) % 4]) for i in range(4)]
        side_lengths = [cv2.norm(side[0] - side[1]) for side in sides]

        # Find the largest side
        largest_side = sides[np.argmax(side_lengths)]

        # Calculate the angle of the largest side with respect to the horizontal axis
        dx = largest_side[1][0] - largest_side[0][0]
        dy = largest_side[1][1] - largest_side[0][1]
        angle = np.arctan2(dy, dx) * 180 / np.pi

        return angle

    @staticmethod
    def square_inside_figure(square, shape):
        x, y, size = square
        square_pixels = shape[y:y + size, x:x + size]
        # Assuming the shape is represented by 255 (white)
        inside_count = np.sum(square_pixels == 255)
        total_count = size * size
        acceptable_transcend = 0.9 * total_count
        return inside_count > acceptable_transcend

    @staticmethod
    def crate_grid(image):
        shape_height, shape_width = image.shape[:2]
        square_size = min(shape_height, shape_width) // 3  # Adjusting the size for visibility
        x_coords = np.arange(0, shape_width, square_size)
        y_coords = np.arange(0, shape_height, square_size)

        return x_coords, y_coords, square_size

    @staticmethod
    def rotate_coordinates(x_coords, y_coords, angle, origin):
        angle_rad = np.deg2rad(angle)
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad), np.cos(angle_rad)]])

        rotated_coords = []
        for x in x_coords:
            for y in y_coords:
                rotated = np.dot(rotation_matrix, np.array([x - origin[0], y - origin[1]])) + origin
                rotated_coords.append((int(rotated[0]), int(rotated[1])))
        return rotated_coords
    @staticmethod
    def find_squares(rotated_coords, square_size, image):
        squares = []
        for x, y in rotated_coords:
            if KitchenTableWithChairs.square_inside_figure((x, y, square_size), image):
                squares.append((x, y, square_size))
        return squares
    @staticmethod
    def find_square_center(squares):
        centers = []
        for square in squares:
            x, y, size = square
            center_x = x + size // 2
            center_y = y + size // 2
            centers.append((center_x, center_y))

        return centers