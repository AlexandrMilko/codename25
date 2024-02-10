import cv2
import numpy as np
import pandas as pd
import os

from PIL import Image
from disruptor.green_screen.find_similar.ssim import (calculate_vanishing_point_by_XiaohuLu,
                                                      find_object_centers,
                                                      calculate_mean_size,
                                                      calculate_iou_for_color,
                                                      convert_to_pil,
                                                      fill_with_white_except,
                                                      convert_to_cv2,
                                                      scale_to_height)
from disruptor.green_screen.find_similar.VanishingPoint.main import manhattan_distance
from disruptor import app
from flask_login import current_user
from flask import url_for
import re
import msvcrt

# TODO Use it for the ML input, with Pipelines too.
# from sklearn.base import BaseEstimator, TransformerMixin
#
# class AttributesAdder(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self
#     def transform(self, X, y=None):
#         pass

def get_all_pairs(numbers: list):
    pairs = []
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            pair = (numbers[i], numbers[j])
            pairs.append(pair)
    return pairs

class Room:
    # BGR, used in segmented images
    window_color = (230, 230, 230)
    door_color = (51, 255, 8)
    floor_color = (50, 50, 80)

    def __init__(self, image_path, is_users_es=False, segmented_users_before=None):
        # If it is not user's uploaded empty space image, we will assign a Before/After to it, to make a pair.
        if not is_users_es:
            # We do this because our dataset consists of images Before, After. As well as Before segmented version
            trio = Room.get_trio(image_path)
            self.segmented_before = trio["segmented_before"]
            self.before = trio["before"]
            self.after = trio["after"]
            # TODO make it calculate these values even with user_empty_space image
            # TODO replace all the hardcode using this class in sdquery.apply_style function
            self.vanishing_point = calculate_vanishing_point_by_XiaohuLu(self.before)
            self.window_centers = sorted(find_object_centers(self.segmented_before, Room.window_color, debug=False))
            self.door_centers = sorted(find_object_centers(self.segmented_before, Room.door_color, debug=False))
        else:
            # WARNING! If you use users_empty space,
            # and you do not specify segmented_users_before,
            # then you have to remember that here we will take the last image ran with segmentation preprocessor.
            # So, if you do so, please run the preprocessor for this image beforehand. Usage Example:
            #   run_preprocessor("seg_ofade20k", es_path)
            #   room_obj = Room(es_path, True) # Just for testing
            # I cannot run preprocessor from here because it will cause circular import error.
            # Maybe I will change architecture in the future, when I refactor the code
            self.before = image_path
            if segmented_users_before is not None:
                self.tmp_segmented_before = segmented_users_before
            else:
                self.tmp_segmented_before = "disruptor" + url_for('static',
                                                                  filename=f'images/{current_user.id}/preprocessed/preprocessed.jpg')
            self.vanishing_point = calculate_vanishing_point_by_XiaohuLu(self.before)
            self.window_centers = sorted(find_object_centers(self.tmp_segmented_before, Room.window_color, debug=False))
            self.door_centers = sorted(find_object_centers(self.tmp_segmented_before, Room.door_color, debug=False))

    @staticmethod
    def get_trio(image_path):
        # We will find its trio in the dataset
        trio = {
            "segmented_before": None,
            "before": None,
            "after": None
        }
        import re
        # Get the room type directory
        room_directory = os.path.dirname(os.path.dirname(image_path))

        image_number = str(
            Room.get_room_number(image_path))
        trio["segmented_before"] = os.path.join(room_directory, "es_segmented/" + image_number + "Before.jpg")
        trio["before"] = os.path.join(room_directory, "original/" + image_number + "Before.jpg")
        trio["after"] = os.path.join(room_directory, "original/" + image_number + "After.jpg")

        return trio

    @property
    def image_size(self):
        with Image.open(self.before) as img:
            width, height = img.size
            return width, height

    @property
    def windows_number(self):
        return len(self.window_centers)

    @property
    def doors_number(self):
        return len(self.door_centers)

    def get_features_values(self):
        # We sort them so that the furthest objects are compared with the furthest and the closest with the closest
        features = [self.vanishing_point, *self.window_centers, *self.door_centers]
        return features

    def measure_similarity(self, other):
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        if (self.windows_number == other.windows_number) and (self.doors_number == other.doors_number):
            features_first = np.array(self.get_features_values())
            features_second = np.array(other.get_features_values())
            first_flattened = np.reshape(features_first, (1, -1))
            second_flattened = np.reshape(features_second, (1, -1))
            similarity_matrix = cosine_similarity(first_flattened, second_flattened)
            return similarity_matrix[0][0]
        else:
            return 0

    def get_vanishing_point_distance(self, other):
        # Since images can have different sizes, we find their mean scale and then rescale the vanishing points and only then calculate the distance
        width1, height1 = self.image_size
        width2, height2 = other.image_size

        mean_width, mean_height = calculate_mean_size(self.image_size, other.image_size)

        vp_x1, vp_y1 = self.vanishing_point
        vp_x2, vp_y2 = other.vanishing_point

        scaled_vp1 = (vp_x1 * (mean_width / width1), vp_y1 * (mean_height / height1))
        scaled_vp2 = (vp_x2 * (mean_width / width2), vp_y2 * (mean_height / height2))

        return int(manhattan_distance(scaled_vp1, scaled_vp2))

    def compare_iou(self, other, rgb_colors=None):  # WARNING, specify RGB colors, because we work with PIL, and it uses RGB
        # Open the image using PIL
        if rgb_colors is None:
            rgb_colors = [Room.floor_color[::-1], Room.window_color[::-1], Room.door_color[::-1]] # We convert it into RGB format
        first_image = Image.open(self.segmented_before)
        second_image = Image.open(other.segmented_before)

        mean_size = calculate_mean_size(first_image.size, second_image.size)
        first_image = first_image.resize(mean_size, Image.Resampling.LANCZOS)
        second_image = second_image.resize(mean_size, Image.Resampling.LANCZOS)
        rgb_colors = [np.array(color) for color in rgb_colors]

        # Calculate IoU
        iou = 0
        for color in rgb_colors:
            iou += calculate_iou_for_color(first_image, second_image, color)

        first_image.close()
        second_image.close()

        return iou

    @staticmethod
    def get_room_number(path):
        return int(re.search(r'\d+', os.path.basename(path)).group())

    @staticmethod
    def ask_if_rooms_similar(room1, room2):
        before1 = room1.before
        before2 = room2.before
        img1 = cv2.imread(before1)
        img2 = cv2.imread(before2)

        # Scale images to have a height of 512 pixels
        img1 = scale_to_height(img1, 512)
        img2 = scale_to_height(img2, 512)

        # Create a blank canvas to display images on the left part of the screen
        canvas = 255 * np.ones((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)

        # Place the scaled images on the canvas
        canvas[:img1.shape[0], :img1.shape[1]] = img1
        canvas[:img2.shape[0], img1.shape[1]:] = img2

        cv2.imshow("Rooms Comparison", canvas)
        print("Are these images similar? (Y/N): ", end='', flush=True)
        user_input = chr(cv2.waitKey(0) & 0xFF)
        cv2.destroyAllWindows()
        return user_input

    @staticmethod
    def is_rooms_entry_saved(dataset_path, room_n1, room_n2):
        try:
            # Load the CSV file into a DataFrame
            df = pd.read_csv(dataset_path)
            # Check if any row matches the specified values for "No1" and "No2"
            row_exists = ((df['No1'] == room_n1) & (df['No2'] == room_n2)).any()
            return row_exists
        except FileNotFoundError:
            # Handle the case where the file doesn't exist
            print(f"File '{dataset_path}' not found.")
            return False

    @staticmethod
    def get_paired_dataframe(seg_directory):
        df = pd.DataFrame(columns=["No1",
                                   "No2",
                                   "Vanishing Point Distance",
                                   "Windows IOU",
                                   "Doors IOU",
                                   "AreSimilar"
                                   ])
        rooms_numbers = [Room.get_room_number(fname) for fname in os.listdir(seg_directory)]
        pairs = get_all_pairs(rooms_numbers)
        dataset_name = "bedroom_dataset.csv"
        for pair in pairs:
            n1, n2 = pair
            if Room.is_rooms_entry_saved(dataset_name, n1, n2):
                continue
            room1 = Room(seg_directory + "\\" + str(n1)+"Before.jpg")
            room2 = Room(seg_directory + "\\" + str(n2)+"Before.jpg")
            vp_dist = room1.get_vanishing_point_distance(room2)
            windows_iou = room1.compare_iou(room2, [Room.window_color[::-1]])
            doors_iou = room1.compare_iou(room2, [Room.door_color[::-1]])
            are_similar = Room.ask_if_rooms_similar(room1, room2)
            row = [n1, n2, vp_dist, windows_iou, doors_iou, are_similar]
            print(row)
            print()
            df = pd.concat([pd.DataFrame(data=[row], columns=df.columns), df], ignore_index=True)
            df.to_csv(dataset_name)

        return df

    @staticmethod
    def get_dataframe(seg_directory):
        df = pd.DataFrame(columns=["No",
                                   "Vanishing Point",
                                   "Window Center 0",
                                   "Window Center 1",
                                   "Door Center 0",
                                   "Door Center 1",
                                   ])
        for fname in os.listdir(seg_directory):
            room = Room(seg_directory + "\\" + fname)

            window_centers = room.window_centers
            for i in range(2 - len(window_centers)):
                window_centers.append(None)

            door_centers = room.door_centers
            for i in range(2 - len(door_centers)):
                door_centers.append(None)

            row = [Room.get_room_number(fname), room.vanishing_point, *window_centers, *door_centers]
            print(row)
            df = pd.concat([pd.DataFrame(data=[row], columns=df.columns), df], ignore_index=True)

        return df

    @staticmethod
    def calculate_means_and_std(df_path):
        result = dict()
        df = pd.read_csv(df_path)
        for col in df.columns:
            if col in ("No", "Unnamed"): continue
            import ast
            mean_x = 0
            mean_y = 0

            for i in df.index:
                try:
                    cell = ast.literal_eval(df[col][i])
                    mean_x += cell[0]
                    mean_y += cell[1]
                except ValueError:
                    continue
            mean_x = round(mean_x / len(df))
            mean_y = round(mean_y / len(df))

            xs = []
            ys = []
            for i in df.index:
                try:
                    if not pd.isna(df[col][i]):
                        cell = ast.literal_eval(df[col][i])
                        xs.append(cell[0])
                        ys.append(cell[1])
                except ValueError as e:
                    pass
            arr_x = np.array(xs)
            arr_y = np.array(ys)
            try:
                std_x = int(np.std(arr_x, ddof=1))
                std_y = int(np.std(arr_y, ddof=1))

                result[col] = {"Mean": (mean_x, mean_y), "Std": (std_x, std_y)}
            except:
                continue

        return result

    def get_normalized_features_values(self, means_std: dict):
        features = self.get_features_values()
        dict_repr = {
            "Vanishing Point": features[0]
        }

        for i in range(self.windows_number):
            dict_repr["Window Center " + str(i)] = features[i + 1] # + 1 for skipping Vanishing Point

        for i in range(self.doors_number):
            dict_repr["Door Center " + str(i)] = features[i + 1 + self.windows_number] # + 1 + self.windows_number for skipping Vanishing Point and Window indexes

        normalized_dict_repr = {}
        for col, (x, y) in dict_repr.items():
            mean_x, mean_y = means_std[col]['Mean']
            std_x, std_y = means_std[col]['Std']

            normalized_x = (x - mean_x) / std_x
            normalized_y = (y - mean_y) / std_y

            normalized_dict_repr[col] = (normalized_x, normalized_y)


        windows = []
        for i in range(self.windows_number):
            windows.append(normalized_dict_repr["Window Center " + str(i)])
        windows = sorted(windows)

        doors = []
        for i in range(self.doors_number):
            doors.append(normalized_dict_repr["Door Center " + str(i)])
        doors = sorted(doors)

        normalized_arr_repr = [normalized_dict_repr["Vanishing Point"], *windows, *doors]

        return normalized_arr_repr