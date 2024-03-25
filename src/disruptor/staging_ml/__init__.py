import pickle

import cv2
import numpy as np
import pandas as pd
import os

from PIL import Image
from disruptor.green_screen.find_similar.ssim import (calculate_vanishing_points_by_XiaohuLu,
                                                      find_object_centers,
                                                      calculate_mean_size,
                                                      calculate_iou_for_colors,
                                                      convert_to_pil,
                                                      fill_with_white_except,
                                                      convert_to_cv2,
                                                      scale_to_height,
                                                      find_biggest_wall_center,
                                                      find_biggest_wall_area)
from disruptor.green_screen.find_similar.VanishingPoint.main import manhattan_distance
from disruptor import app
from flask_login import current_user
from flask import url_for
import re

from sklearn.base import BaseEstimator, TransformerMixin


class AttributesAdder(BaseEstimator, TransformerMixin):
    # training_attributes = ["Windows Doors IOU", "Floor IOU", "Windows IOU", "Doors IOU", "VPD Right", "VPD Left", "VPD Vertical", "Biggest Wall Center Distance"]
    training_attributes = ["Windows Doors IOU", "VPD Left", "VPD Right", "Floor IOU"]
    # training_attributes = ["VPD Right", "VPD Left", "VPD Vertical", "Windows IOU", "Doors IOU", "Floor IOU",
    #                        "Windows Doors IOU", "Biggest Wall Center Distance", "Biggest Wall Center Area Diff"]

    def __init__(self, segmented_directory=None, update_vp=False, update_iou=False, update_wall_center=False,
                 length_thresh=60, focal_length=500):
        self.segmented_directory = segmented_directory
        self.update_vp = update_vp
        self.update_iou = update_iou
        self.update_wall_center = update_wall_center
        self.length_thresh = length_thresh
        self.focal_length = focal_length

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(self.length_thresh, self.focal_length, " LENGTH THRESH and FOCAL LENGTH")
        if self.segmented_directory:
            for index, row in X.iterrows():
                # print(f"{index}/{len(X)} rooms processed by AttributesAdder")
                room1 = Room(self.segmented_directory + "\\" + str(row["No1"]) + "Before.jpg",
                             length_thresh=self.length_thresh, focal_length=self.focal_length)
                room2 = Room(self.segmented_directory + "\\" + str(row["No2"]) + "Before.jpg",
                             length_thresh=self.length_thresh, focal_length=self.focal_length)
                # print("Are {}, {} rooms similar?".format(row["No1"], row["No2"]), row["AreSimilar"])
                if self.update_vp:
                    vp_dist = room1.get_vanishing_points_distances(room2)
                    X.at[index, 'VPD Right'] = vp_dist[0]
                    X.at[index, 'VPD Left'] = vp_dist[1]
                    X.at[index, 'VPD Vertical'] = vp_dist[2]
                if self.update_iou:
                    windows_iou = room1.compare_iou(room2, [Room.window_color[::-1]])
                    doors_iou = room1.compare_iou(room2, [Room.door_color[::-1]])
                    floor_iou = room1.compare_iou(room2, [Room.floor_color[::-1]])
                    windows_doors_iou = room1.compare_iou(room2, [Room.door_color[::-1], Room.window_color[::-1]])
                    X.at[index, 'Windows IOU'] = windows_iou
                    X.at[index, 'Doors IOU'] = doors_iou
                    X.at[index, 'Floor IOU'] = floor_iou
                    X.at[index, 'Windows Doors IOU'] = windows_doors_iou
                if self.update_wall_center:
                    wall_center_dist = room1.get_biggest_walls_distance(room2)
                    # biggest_walls_area_difference = room1.get_biggest_walls_area_difference(room2)
                    X.at[index, 'Biggest Wall Center Distance'] = wall_center_dist
                    # X.at[index, 'Biggest Wall Center Area Diff'] = biggest_walls_area_difference

        return X


class DataframeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


pair_index = 0


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

    def __init__(self, image_path, is_users_es=False, segmented_users_before=None, length_thresh=60, focal_length=500):
        # If it is not user's uploaded empty space image, we will assign a Before/After to it, to make a pair.
        if not is_users_es:
            # We do this because our dataset consists of images Before, After. As well as Before segmented version
            trio = Room.get_trio(image_path)
            self.segmented_before = trio["segmented_before"]
            self.before = trio["before"]
            self.after = trio["after"]
            self.length_thresh = length_thresh
            self.focal_length = focal_length
            # TODO make it calculate these values even with user_empty_space image
            # TODO replace all the hardcode using this class in sdquery.apply_style function
            self.vanishing_points = calculate_vanishing_points_by_XiaohuLu(self.before, self.length_thresh,
                                                                           self.focal_length)
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
                self.segmented_before = segmented_users_before
            else:
                self.segmented_before = "disruptor" + url_for('static',
                                                              filename=f'images/{current_user.id}/preprocessed/preprocessed.jpg')
            self.length_thresh = length_thresh
            self.focal_length = focal_length
            self.vanishing_points = calculate_vanishing_points_by_XiaohuLu(self.before, self.length_thresh,
                                                                           self.focal_length)
            self.window_centers = sorted(find_object_centers(self.segmented_before, Room.window_color, debug=False))
            self.door_centers = sorted(find_object_centers(self.segmented_before, Room.door_color, debug=False))

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
        features = [*self.vanishing_points, *self.window_centers, *self.door_centers]
        return features

    # def measure_similarity(self, other):
    #     from sklearn.metrics.pairwise import cosine_similarity
    #     import numpy as np
    #     if (self.windows_number == other.windows_number) and (self.doors_number == other.doors_number):
    #         features_first = np.array(self.get_features_values())
    #         features_second = np.array(other.get_features_values())
    #         first_flattened = np.reshape(features_first, (1, -1))
    #         second_flattened = np.reshape(features_second, (1, -1))
    #         similarity_matrix = cosine_similarity(first_flattened, second_flattened)
    #         return similarity_matrix[0][0]
    #     else:
    #         return 0
    def measure_similarity(self, other):
        from sklearn.pipeline import Pipeline
        pair_df = pd.DataFrame(columns=['No1', 'No2'])
        pair = dict()
        pair['No1'] = Room.get_room_number(self.before)
        pair['No2'] = Room.get_room_number(other.before)
        # TODO change path, along with your dataset and clf.pkl
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        pipeline = Pipeline([
            ('selector', DataframeSelector(AttributesAdder.training_attributes)),
            ('scaler', scaler)
        ])
        print("Started")
        vp_dist = self.get_vanishing_points_distances(other)
        pair['VPD Right'] = vp_dist[0]
        pair['VPD Left'] = vp_dist[1]
        pair['VPD Vertical'] = vp_dist[2]
        windows_iou = self.compare_iou(other, [Room.window_color[::-1]])
        doors_iou = self.compare_iou(other, [Room.door_color[::-1]])
        floor_iou = self.compare_iou(other, [Room.floor_color[::-1]])
        windows_doors_iou = self.compare_iou(other, [Room.door_color[::-1], Room.window_color[::-1]])
        pair['Windows IOU'] = windows_iou
        pair['Doors IOU'] = doors_iou
        pair['Floor IOU'] = floor_iou
        pair['Windows Doors IOU'] = windows_doors_iou
        wall_center_dist = self.get_biggest_walls_distance(other)
        pair['Biggest Wall Center Distance'] = wall_center_dist

        pair_df = pd.concat([pair_df, pd.DataFrame([pair])], ignore_index=True)
        pair_prepared = pipeline.transform(pair_df)
        model_path = 'clf.pkl'
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
            score = clf.decision_function(pair_prepared)
            print(pair["No1"], pair["No2"], score)
            return score

    def get_biggest_walls_distance(self, other):
        width1, height1 = self.image_size
        width2, height2 = other.image_size

        mean_width, mean_height = calculate_mean_size(self.image_size, other.image_size)
        x1, y1 = find_biggest_wall_center(self.segmented_before)
        x2, y2 = find_biggest_wall_center(other.segmented_before)

        scaled_pos_first = (x1 * (mean_width / width1), y1 * (mean_height / height1))
        scaled_pos_second = (x2 * (mean_width / width2), y2 * (mean_height / height2))

        return int(manhattan_distance(scaled_pos_first, scaled_pos_second))

    def get_biggest_walls_area_difference(self, other):
        width1, height1 = self.image_size
        width2, height2 = other.image_size

        mean_width, mean_height = calculate_mean_size(self.image_size, other.image_size)
        area1 = find_biggest_wall_area(self.segmented_before)
        area2 = find_biggest_wall_area(other.segmented_before)

        scaled_area1 = area1 * (mean_width / width1) * (mean_height / height1)
        scaled_area2 = area2 * (mean_width / width2) * (mean_height / height2)
        return abs(scaled_area1 - scaled_area2)

    def get_vanishing_points_distances(self, other):
        # Since images can have different sizes, we find their mean scale and then rescale the vanishing points and only then calculate the distance
        width1, height1 = self.image_size
        width2, height2 = other.image_size

        mean_width, mean_height = calculate_mean_size(self.image_size, other.image_size)

        [[vp1_x1, vp1_y1], [vp1_x2, vp1_y2], [vp1_x3, vp1_y3]] = self.vanishing_points
        [[vp2_x1, vp2_y1], [vp2_x2, vp2_y2], [vp2_x3, vp2_y3]] = other.vanishing_points

        scaled_vp_right_first = (vp1_x1 * (mean_width / width1), vp1_y1 * (mean_height / height1))
        scaled_vp_left_first = (vp1_x2 * (mean_width / width1), vp1_y2 * (mean_height / height1))
        scaled_vp_vertical_first = (vp1_x3 * (mean_width / width1), vp1_y3 * (mean_height / height1))

        scaled_vp_right_second = (vp2_x1 * (mean_width / width2), vp2_y1 * (mean_height / height2))
        scaled_vp_left_second = (vp2_x2 * (mean_width / width2), vp2_y2 * (mean_height / height2))
        scaled_vp_vertical_second = (vp2_x3 * (mean_width / width2), vp2_y3 * (mean_height / height2))

        mean_vp_right = ((scaled_vp_right_first[0] + scaled_vp_right_second[0]) // 2,
                         (scaled_vp_right_first[1] + scaled_vp_right_second[1]) // 2
                         )
        mean_vp_left = ((scaled_vp_left_first[0] + scaled_vp_left_second[0]) // 2,
                        (scaled_vp_left_first[1] + scaled_vp_left_second[1]) // 2
                        )
        mean_vp_vertical = ((scaled_vp_vertical_first[0] + scaled_vp_vertical_second[0]) // 2,
                            (scaled_vp_vertical_first[1] + scaled_vp_vertical_second[1]) // 2
                            )
        mean_dist_right = int(manhattan_distance(mean_vp_right, [0, 0]))
        mean_dist_left = int(manhattan_distance(mean_vp_left, [0, 0]))
        mean_dist_vertical = int(manhattan_distance(mean_vp_vertical, [0, 0]))

        right_relative_dist = int(manhattan_distance(scaled_vp_right_first, scaled_vp_right_second)) / mean_dist_right
        left_relative_dist = int(manhattan_distance(scaled_vp_left_first, scaled_vp_left_second)) / mean_dist_left
        vertical_relative_dist = int(
            manhattan_distance(scaled_vp_vertical_first, scaled_vp_vertical_second)) / mean_dist_vertical

        # DEBUG
        # Room.show_pair_and_vp(self, other, scaled_vp_right_first, scaled_vp_right_second, "Right VP comparison")
        # Room.show_pair_and_vp(self, other, scaled_vp_left_first, scaled_vp_left_second, "Left VP comparison")

        # from sklearn.metrics.pairwise import cosine_similarity

        # return [cosine_similarity([scaled_vp_right_first], [scaled_vp_right_second])[0][0],
        #         cosine_similarity([scaled_vp_left_first], [scaled_vp_left_second])[0][0],
        #         cosine_similarity([scaled_vp_vertical_first], [scaled_vp_vertical_second])[0][0]]

        # return [int(manhattan_distance(scaled_vp_right_first, scaled_vp_right_second)),
        #         int(manhattan_distance(scaled_vp_left_first, scaled_vp_left_second)),
        #         int(manhattan_distance(scaled_vp_vertical_first, scaled_vp_vertical_second))]
        return right_relative_dist, left_relative_dist, vertical_relative_dist

    def compare_iou(self, other,
                    rgb_colors=None):  # WARNING, specify RGB colors, because we work with PIL, and it uses RGB
        # Open the image using PIL
        if rgb_colors is None:
            rgb_colors = [Room.floor_color[::-1], Room.window_color[::-1],
                          Room.door_color[::-1]]  # We convert it into RGB format
        first_image = Image.open(self.segmented_before)
        second_image = Image.open(other.segmented_before)

        mean_size = calculate_mean_size(first_image.size, second_image.size)
        first_image = first_image.resize(mean_size, Image.Resampling.LANCZOS)
        second_image = second_image.resize(mean_size, Image.Resampling.LANCZOS)

        # Calculate IoU
        iou = calculate_iou_for_colors(first_image, second_image, rgb_colors)

        first_image.close()
        second_image.close()

        return iou

    @staticmethod
    def get_room_number(path):
        return int(re.search(r'\d+', os.path.basename(path)).group())

    # @staticmethod
    # def show_pair_and_vp(room1, room2, vp1, vp2, test_name):
    #     before1 = room1.before
    #     before2 = room2.before
    #     img1 = cv2.imread(before1)
    #     img2 = cv2.imread(before2)
    #
    #     mean_width, mean_height = calculate_mean_size(room1.image_size, room2.image_size)
    #
    #     img1 = cv2.resize(img1, (mean_width // 2, mean_height // 2))
    #     img2 = cv2.resize(img2, (mean_width // 2, mean_height // 2))
    #
    #     # Set transparency level
    #     alpha = 0.5
    #     # Create a transparent overlay
    #     overlay = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
    #
    #     # Draw vanishing points
    #     cv2.circle(overlay, [int(vp1[0] // 2), int(vp1[1] // 2)], 5, (0, 255, 0), -1)
    #     cv2.circle(overlay, [int(vp2[0] // 2), int(vp2[1] // 2)], 5, (0, 0, 255), -1)
    #
    #     n1 = Room.get_room_number(before1)
    #     n2 = Room.get_room_number(before2)
    #
    #     cv2.imshow(f"{test_name}: {n1} {n2}", overlay)
    #     user_input = chr(cv2.waitKey(0) & 0xFF)
    #     cv2.destroyAllWindows()
    #     return user_input

    @staticmethod
    def show_pair_and_vp(room1, room2, vp1, vp2, test_name):
        before1 = room1.before
        before2 = room2.before
        img1 = cv2.imread(before1)
        img2 = cv2.imread(before2)

        # Calculate the mean size
        mean_width, mean_height = calculate_mean_size(room1.image_size, room2.image_size)

        # Resize images to half of their mean size
        img1 = cv2.resize(img1, (mean_width // 2, mean_height // 2))
        img2 = cv2.resize(img2, (mean_width // 2, mean_height // 2))

        # Add 250 pixels of extension to each side
        extension = 125

        # Calculate the dimensions of the canvas
        canvas_width = mean_width + 2 * extension
        canvas_height = mean_height + 2 * extension

        # Create the canvas
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Calculate the positions to place the images at the center
        x_offset1 = (canvas_width - img1.shape[1]) // 2
        y_offset1 = (canvas_height - img1.shape[0]) // 2
        x_offset2 = (canvas_width - img2.shape[1]) // 2
        y_offset2 = (canvas_height - img2.shape[0]) // 2

        # Overlay images on the canvas with alpha = 0.5
        alpha = 0.5
        overlay1 = canvas.copy()
        overlay2 = canvas.copy()

        overlay1[y_offset1:y_offset1 + img1.shape[0], x_offset1:x_offset1 + img1.shape[1]] = img1
        overlay2[y_offset2:y_offset2 + img2.shape[0], x_offset2:x_offset2 + img2.shape[1]] = img2

        overlay = cv2.addWeighted(overlay1, alpha, overlay2, 1 - alpha, 0)

        # Draw vanishing points on the canvas
        vp1_x = vp1[0] + x_offset1 - extension
        vp1_y = vp1[1] + y_offset1 - extension
        vp2_x = vp2[0] + x_offset2 - extension
        vp2_y = vp2[1] + y_offset2 - extension

        cv2.circle(overlay, (int(vp1_x), int(vp1_y)), 5, (0, 255, 0), -1)
        cv2.circle(overlay, (int(vp2_x), int(vp2_y)), 5, (0, 0, 255), -1)

        n1 = Room.get_room_number(before1)
        n2 = Room.get_room_number(before2)

        cv2.imshow(f"{test_name}: {n1} {n2}", overlay)
        user_input = chr(cv2.waitKey(0) & 0xFF)
        cv2.destroyAllWindows()
        return user_input

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
    def create_paired_dataframe(seg_directory):
        df = pd.DataFrame(columns=["No1",
                                   "No2",
                                   "AreSimilar"
                                   ])
        rooms_numbers = [Room.get_room_number(fname) for fname in os.listdir(seg_directory)]
        pairs = get_all_pairs(rooms_numbers)
        dataset_name = "bedroom_dataset.csv"
        if os.path.isfile(dataset_name):
            df = pd.read_csv(dataset_name)
        i = 0
        for pair in pairs:
            n1, n2 = pair
            if Room.is_rooms_entry_saved(dataset_name, n1, n2):
                i += 1
                continue
            room1 = Room(seg_directory + "\\" + str(n1) + "Before.jpg")
            room2 = Room(seg_directory + "\\" + str(n2) + "Before.jpg")
            are_similar = Room.ask_if_rooms_similar(room1, room2)
            row = [n1, n2, are_similar]
            i += 1
            print(row)
            print(f"Pair number: {i}/{len(pairs)}")
            print()
            df = pd.concat([pd.DataFrame(data=[row], columns=df.columns), df], ignore_index=True)
            df.to_csv(dataset_name, index=False)

        return df

    @staticmethod
    def get_dataframe(seg_directory):
        df = pd.DataFrame(columns=["No",
                                   "Vanishing Point Right",
                                   "Vanishing Point Left",
                                   "Vanishing Point Vertical",
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

            row = [Room.get_room_number(fname), *room.vanishing_points, *window_centers, *door_centers]
            print(row)
            df = pd.concat([pd.DataFrame(data=[row], columns=df.columns), df], ignore_index=True)

        return df

    @staticmethod
    def save_stratified_dataset(dataset_path):
        from sklearn.model_selection import StratifiedShuffleSplit
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        df = pd.read_csv(dataset_path)
        for train_index, test_index in split.split(df, df["AreSimilar"]):
            strat_train_set = df.loc[train_index]
            strat_test_set = df.loc[test_index]

        strat_train_set.to_csv("train.csv", index=False)
        strat_test_set.to_csv("test.csv", index=False)

    @staticmethod
    def visualize_dataset(dataset_path, model_path='clf.pkl'):
        df = pd.read_csv(dataset_path)

        import matplotlib.pyplot as plt
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df[["VPD Right",
            "VPD Left",
            "VPD Vertical",
            "Windows IOU",
            "Doors IOU",
            "Floor IOU",
            "Windows Doors IOU"]] = scaler.fit_transform(df[[
            "VPD Right",
            "VPD Left",
            "VPD Vertical",
            "Windows IOU",
            "Doors IOU",
            "Floor IOU",
            "Windows Doors IOU"]])

        # 3D VISUALIZATION

        color_map = {'y': 'green', 'n': 'red'}
        df['Color'] = df['AreSimilar'].map(color_map)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df["VPD Left"], df["VPD Right"], df["Windows Doors IOU"],
                   c=df['Color'], marker='o', alpha=0.1)

        # for index, row in df.iterrows():
        #     # Get the IDs
        #     if row["AreSimilar"] == "n":
        #         no1 = str(row["No1"])
        #         no2 = str(row["No2"])
        #
        #         # Label each point with "No1_No2"
        #         ax.text(row["VPD Left"], row["VPD Vertical"], row["Windows IOU"], "{}_{}".format(no1, no2), color='black')

        # Set labels and title
        ax.set_xlabel('VPD Left')
        ax.set_ylabel('VPD Right')
        ax.set_zlabel('Windows Doors IOU')
        ax.set_title('3D Visualization of DataFrame')

        # CORRELATION

        label_map = {'y': 1, 'n': 0}
        df['Label'] = df['AreSimilar'].map(label_map)
        df.drop(columns=['AreSimilar'], inplace=True)
        df.drop(columns=['Color'], inplace=True)
        corr_matrix = df.corr()
        print(corr_matrix['Label'].sort_values(ascending=False))

        # SCATTER MATRIX
        from pandas.plotting import scatter_matrix

        attributes = [*AttributesAdder.training_attributes, "Label"]
        scatter_matrix(df[attributes], figsize=(12, 8))

        plt.show()

    @staticmethod
    def visualize_polynomial_features(dataset_path, degree=2):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        df = pd.read_csv(dataset_path)
        for i in range(len(AttributesAdder.training_attributes)):
            pipeline = Pipeline([
                ('selector', DataframeSelector([AttributesAdder.training_attributes[i]])),
                ('scaler', StandardScaler())
            ])
            X1D = pipeline.fit_transform(df)
            X2D = np.c_[X1D, X1D ** degree]
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(df["AreSimilar"])

            # VISUALIZE ATTRIBUTE
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 3))

            plt.subplot(121)
            plt.grid(True, which='both')
            plt.axhline(y=0, color='k')
            plt.plot(X1D[:, 0][y == 0], np.zeros(486), "bs")
            plt.plot(X1D[:, 0][y == 1], np.zeros(322), "g^")
            plt.xlabel(r"$d_1$" + AttributesAdder.training_attributes[i], fontsize=20)

            plt.subplot(122)
            plt.grid(True, which='both')
            plt.axhline(y=0, color='k')
            plt.axvline(x=0, color='k')
            plt.plot(X2D[:, 0][y == 0], X2D[:, 1][y == 0], "bs")
            plt.plot(X2D[:, 0][y == 1], X2D[:, 1][y == 1], "g^")
            plt.xlabel(r"$d_1$" + AttributesAdder.training_attributes[i], fontsize=20)
            plt.ylabel(r"$d_2$" + AttributesAdder.training_attributes[i], fontsize=20, rotation=0)

            plt.subplots_adjust(right=1)

            plt.show()

    @staticmethod
    def plot_predictions(dataset_path, model_path="clf.pkl"):
        with open(model_path, 'rb') as f:
            import matplotlib.pyplot as plt
            clf = pickle.load(f)
            X, y = Room.prepare_dataset(dataset_path)
            x0min, x0max = X[:, 0].min(), X[:, 0].max()
            x1min, x1max = X[:, 1].min(), X[:, 1].max()
            x2min, x2max = X[:, 2].min(), X[:, 2].max()
            x3min, x3max = X[:, 3].min(), X[:, 3].max()
            steps = 8
            x0s = np.linspace(x0min, x0max, steps)
            x1s = np.linspace(x1min, x1max, steps)
            x2s = np.linspace(x2min, x2max, steps)
            x3s = np.linspace(x3min, x3max, steps)

            x0, x1, x2, x3 = np.meshgrid(x0s, x1s, x2s, x3s)

            X = np.c_[x0.ravel(), x1.ravel(), x2.ravel(), x3.ravel()]

            y_pred = clf.predict(X)
            y_pred = y_pred.reshape(x0.shape)

            y_decision = clf.decision_function(X)
            y_decision = y_decision.reshape(x0.shape)

            plt.contourf(x0[:, :, 0, 0], x1[:, :, 0, 0], y_pred[:, :, 0, 0], cmap=plt.cm.brg, alpha=0.2)
            plt.contourf(x0[:, :, 0, 0], x1[:, :, 0, 0], y_decision[:, :, 0, 0], cmap=plt.cm.brg, alpha=0.1)
            plt.show()

    @staticmethod
    def visualize_all_vanishing_points(seg_directory, vp_type):
        if vp_type.lower() == "right":
            type_index = 0
        elif vp_type.lower() == "left":
            type_index = 1
        elif vp_type.lower() == "vertical":
            type_index = 2
        else:
            raise Exception("Please, specify vanishing point type: right, left or vertical")
        rooms_numbers = [Room.get_room_number(fname) for fname in os.listdir(seg_directory)]
        df = pd.DataFrame(columns=["No", "Width", "Height", "VP X", "VP Y"])
        for No in rooms_numbers:
            print(No, " Added to df")
            room = Room(seg_directory + "\\" + str(No) + "Before.jpg")
            df = pd.concat([pd.DataFrame(data=[[No,
                                                room.image_size[0],
                                                room.image_size[1],
                                                room.vanishing_points[type_index][0],
                                                room.vanishing_points[type_index][1]]],
                                         columns=["No", "Width", "Height", "VP X", "VP Y"]), df], ignore_index=True)
        mean_width = int(df["Width"].mean())
        mean_height = int(df["Height"].mean())

        df["Scaled VP X"] = df["VP X"] * (mean_width / df["Width"])
        df["Scaled VP Y"] = df["VP Y"] * (mean_height / df["Height"])
        print(mean_width, mean_height, " - MEAN SIZE")
        print(df.head())

        import matplotlib.pyplot as plt
        fig = plt.figure()
        vp_plot = fig.add_subplot(111)
        vp_plot.scatter(df["Scaled VP X"], df["Scaled VP Y"], c='green', marker='o', alpha=0.1)

        for i, label in enumerate(df["No"]):
            vp_plot.annotate(label, (df["Scaled VP X"][i], df["Scaled VP Y"][i]), textcoords="offset points",
                             xytext=(0, 10), ha='center')

        plt.show()

    @staticmethod
    def visualize_pairs(seg_directory, paired_dataset_path, vp_type):
        if vp_type.lower() == "right":
            type_index = 0
        elif vp_type.lower() == "left":
            type_index = 1
        elif vp_type.lower() == "vertical":
            type_index = 2
        else:
            raise Exception("Please, specify vanishing point type: right, left or vertical")

        rooms_numbers = [Room.get_room_number(fname) for fname in os.listdir(seg_directory)]
        img_and_vp_df = pd.DataFrame(columns=["No", "Width", "Height", "VP X", "VP Y"])
        for No in rooms_numbers:
            print(No, " Added to img_and_vp_df")
            room = Room(seg_directory + "\\" + str(No) + "Before.jpg")
            img_and_vp_df = pd.concat([pd.DataFrame(data=[[No,
                                                           room.image_size[0],
                                                           room.image_size[1],
                                                           room.vanishing_points[type_index][0],
                                                           room.vanishing_points[type_index][1]]],
                                                    columns=["No", "Width", "Height", "VP X", "VP Y"]), img_and_vp_df],
                                      ignore_index=True)
        mean_width = int(img_and_vp_df["Width"].mean())
        mean_height = int(img_and_vp_df["Height"].mean())

        img_and_vp_df["Scaled VP X"] = img_and_vp_df["VP X"] * (mean_width / img_and_vp_df["Width"])
        img_and_vp_df["Scaled VP Y"] = img_and_vp_df["VP Y"] * (mean_height / img_and_vp_df["Height"])

        paired_df = pd.read_csv(paired_dataset_path)
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 6))
        vp_plot = fig.add_subplot(111, projection='3d')

        def on_key(event):
            global pair_index
            if event.key == 'enter':
                pair_index = (pair_index + 1) % len(paired_df)  # Cycle through samples
                update_plot()

        fig.canvas.mpl_connect('key_press_event', on_key)

        def update_plot():
            global pair_index
            pair_row = paired_df.iloc[pair_index]

            vp1_x = img_and_vp_df[img_and_vp_df["No"] == pair_row["No1"]]["Scaled VP X"]
            vp2_x = img_and_vp_df[img_and_vp_df["No"] == pair_row["No2"]]["Scaled VP X"]

            vp1_y = img_and_vp_df[img_and_vp_df["No"] == pair_row["No1"]]["Scaled VP Y"]
            vp2_y = img_and_vp_df[img_and_vp_df["No"] == pair_row["No2"]]["Scaled VP Y"]

            windows_iou = pair_row["Windows IOU"]

            vp_plot.clear()

            vp_plot.set_xlim([-1500, 1500])
            vp_plot.set_ylim([-1000, 1000])
            vp_plot.set_zlim([0, 0.3])
            color = 'green' if pair_row["AreSimilar"] == 'y' else 'red'
            vp_plot.scatter(vp1_x, vp1_y, windows_iou, c=color)
            vp_plot.scatter(vp2_x, vp2_y, windows_iou, c=color)
            print(pair_row["No1"], pair_row["No2"], " PAIR")

            # Set labels
            vp_plot.set_xlabel('Scaled VP X')
            vp_plot.set_ylabel('Scaled VP Y')
            vp_plot.set_zlabel('Windows IOU')
            vp_plot.set_title(f'Sample {pair_index + 1}')
            vp_plot.legend()
            plt.draw()

        update_plot()

        plt.show()

    @staticmethod
    def visualize_box_plots(dataset_path):
        import matplotlib.pyplot as plt
        df = pd.read_csv(dataset_path)
        df_true = df[df['AreSimilar'] == 'y']
        df_false = df[df['AreSimilar'] == 'n']

        fig = plt.figure()
        box_plot_true = fig.add_subplot(121)
        box_plot_false = fig.add_subplot(122)
        box_plot_true.set_title("Similar")
        box_plot_false.set_title("Not Similar")

        df_true[["VPD Right",
                 "VPD Left",
                 "VPD Vertical",
                 "Windows IOU", "Doors IOU", "Floor IOU"]].boxplot(ax=box_plot_true, sym='g', showfliers=False)
        df_false[["VPD Right",
                  "VPD Left",
                  "VPD Vertical",
                  "Windows IOU", "Doors IOU", "Floor IOU"]].boxplot(ax=box_plot_false, sym='g', showfliers=False)

        # df_true[["Windows IOU", "Doors IOU"]].boxplot(ax=box_plot_true)
        # df_false[["Windows IOU", "Doors IOU"]].boxplot(ax=box_plot_false)
        plt.show()

    # Use AttributesAdder instead
    # @staticmethod
    # def update_dataset(segmented_directory, dataset_path):
    #     # We use this method to recalculate the columns in the dataset, just for testing and easier tinkering with
    #     df = pd.read_csv(dataset_path)
    #     # df.drop(columns=["VPD Right",
    #     #                  "VPD Left",
    #     #                  "VPD Vertical",
    #     #                  "Windows IOU", "Doors IOU"], inplace=True)
    #     df.drop(columns=["Vanishing Point Distance"], inplace=True)
    #     for index, row in df.iterrows():
    #         room1 = Room(segmented_directory + "\\" + str(row["No1"]) + "Before.jpg")
    #         room2 = Room(segmented_directory + "\\" + str(row["No2"]) + "Before.jpg")
    #         vp_dist = room1.get_vanishing_points_distances(room2)
    #         # windows_iou = room1.compare_iou(room2, [Room.window_color[::-1]])
    #         # doors_iou = room1.compare_iou(room2, [Room.door_color[::-1]])
    #         df.at[index, 'VPD Right'] = vp_dist[0]
    #         df.at[index, 'VPD Left'] = vp_dist[1]
    #         df.at[index, 'VPD Vertical'] = vp_dist[2]
    #         # df.at[index, 'Windows IOU'] = windows_iou
    #         # df.at[index, 'Doors IOU'] = doors_iou
    #
    #     df.to_csv(dataset_path, index=False)

    @staticmethod
    def prepare_dataset(dataset_path):
        # We use it to process dataframe through our pipeline
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        df = pd.read_csv(dataset_path)

        pipeline = Pipeline([
            ('attr_adder', AttributesAdder()),
            ('selector', DataframeSelector(AttributesAdder.training_attributes)),
            ('scaler', StandardScaler())
        ])
        data = pipeline.fit_transform(df)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(pipeline.named_steps['scaler'], f)

        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(df["AreSimilar"])

        return data, labels

    @staticmethod
    def train(dataset_path="train.csv", model_path='clf.pkl'):
        training_data, training_labels = Room.prepare_dataset(dataset_path)

        from sklearn.svm import SVC
        # C = 1000, coef0 = 4, degree = 2, kernel = 'poly'
        # C = 0.1, gamma = 1, kernel = 'rbf'
        # C = 1, gamma = 0.001, kernel = 'rbf'
        # C=0.1, coef0=1, degree=2, kernel='poly' # This one
        # C = 0.975, coef0 = 1, degree = 2, kernel = 'poly'
        # C = 0.43, coef0 = 1, degree = 2, kernel = 'poly'

        # C=0.05, coef0=3, kernel='poly'
        clf = SVC(C=0.05, coef0=3, kernel='poly')
        clf.fit(training_data, training_labels)

        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)

    @staticmethod
    def train_dt(dataset_path="train.csv", model_path="clf.pkl"):
        training_data, training_labels = Room.prepare_dataset(dataset_path)

        from sklearn.tree import DecisionTreeClassifier, export_graphviz
        clf = DecisionTreeClassifier(max_depth=2)
        clf.fit(training_data, training_labels)
        export_graphviz(
            clf,
            out_file="tree.dot"
        )

        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)

    @staticmethod
    def test(dataset_path="test.csv", model_path='clf.pkl'):
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
            test_data, test_labels = Room.prepare_dataset(dataset_path)
            accuracy = clf.score(test_data, test_labels)
            print(accuracy, "ACCURACY")

    @staticmethod
    def find_best_ml_parameters(dataset_path="train.csv"):
        from sklearn.model_selection import GridSearchCV
        param_grid = [{"attr_adder__length_thresh": [60],
                       "attr_adder__focal_length": [100, 500, 1150, 1300, 1450, 1600, 1750, 1900, 2150, 2300, 2450, 2600,
                                                    2750, 2900],
                       "clf__C": [0.05],
                       "clf__degree": [2],
                       "clf__coef0": [3],
                       "clf__kernel": ['poly']
                       }]
        # param_grid = [{
        #     "clf__C": [50, 75, 100, 125, 150, 175, 200],
        #     "clf__gamma": [0.75, 0.9, 1, 1.1, 1.2, 1.3],
        #     "clf__kernel": ['rbf']
        # }]
        # param_grid = [{
        #     "clf__C": [0.1, 1, 10, 100, 500, 1000, 750],
        #     "clf__degree": [None, 2, 3],
        #     "clf__coef0": [1, 2, 3, 4, 5],
        #     "clf__kernel": ['poly']
        # }]
        # param_grid = [{
        #     "clf__C": [i/200 for i in range(0, 201)],
        #     "clf__degree": [None, 2, 3],
        #     "clf__coef0": [1, 2, 3],
        #     "clf__kernel": ['poly']
        # }]

        # param_grid = [{
        #     "clf__max_depth": [2, 3, 4, 5, 6, 7, 8]
        # }]

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        segmented_dir = r"C:\Users\Sasha\Desktop\Projects\codename25\src\disruptor\green_screen\find_similar\dataset\bedroom\es_segmented"
        pipeline = Pipeline([
            ('attr_adder', AttributesAdder(segmented_directory=segmented_dir, update_vp=True)),
            ('selector', DataframeSelector(AttributesAdder.training_attributes)),
            ('scaler', StandardScaler()),
            ('clf', SVC())
        ])

        data = pipeline.named_steps['attr_adder'].fit_transform(pd.read_csv(dataset_path))
        _, labels = Room.prepare_dataset(dataset_path)

        grid_search = GridSearchCV(pipeline, param_grid, scoring='f1', cv=3)
        grid_search.fit(data, labels)

        cv_results = grid_search.cv_results_
        i = 0

        for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):
            print("Mean Score:", mean_score, "Parameters:", params)
            i += 1
        print(i, " CV RESULTS LENGTH")

        print(grid_search.best_estimator_)
        print(grid_search.best_score_, "BEST SCORE")

    @staticmethod
    def perform_cross_val_score(train_dataset_path="train.csv", model_path="clf.pkl"):
        from sklearn.model_selection import cross_val_score, cross_val_predict
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
            training_data, training_labels = Room.prepare_dataset(train_dataset_path)
            # print(cross_val_score(clf, training_data, training_labels, cv=3, scoring="accuracy"))
            training_predict = cross_val_predict(clf, training_data, training_labels, cv=3)

            from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
            print(confusion_matrix(training_labels, training_predict))
            print(precision_score(training_labels, training_predict), "% of cases of my SIMILAR labeling are true")
            print(recall_score(training_labels, training_predict), "% of SIMILAR rooms I can notice")
            print(f1_score(training_labels, training_predict), " F1 SCORE")

    @staticmethod
    def draw_precision_recall_vs_threshold(train_dataset_path="train.csv", model_path="clf.pkl"):
        with open(model_path, "rb") as f:
            clf = pickle.load(f)

            training_data, training_labels = Room.prepare_dataset(train_dataset_path)
            from sklearn.model_selection import cross_val_predict
            training_predict = cross_val_predict(clf, training_data, training_labels, cv=3, method="decision_function")

            from sklearn.metrics import precision_recall_curve
            precisions, recalls, thresholds = precision_recall_curve(training_labels, training_predict)

            import matplotlib.pyplot as plt
            plt.plot(thresholds, precisions[:-1], 'b--', label="Precision")
            plt.plot(thresholds, recalls[:-1], 'g-', label="Recall")
            plt.xlabel("Threshold")
            plt.legend(loc='center')
            plt.ylim([0, 1])
            plt.show()

    @staticmethod
    def draw_roc_curve(train_dataset_path="train.csv", model_path="clf.pkl"):
        from sklearn.model_selection import cross_val_predict
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
            training_data, training_labels = Room.prepare_dataset(train_dataset_path)
            training_predict = cross_val_predict(clf, training_data, training_labels, cv=3, method="decision_function")

            from sklearn.metrics import roc_curve, roc_auc_score
            fpr, tpr, thresholds = roc_curve(training_labels, training_predict)

            import matplotlib.pyplot as plt
            plt.plot(fpr, tpr, linewidth=2, label=None)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.axis((0, 1, 0, 1))
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            print("ROC AUC Score: ", roc_auc_score(training_labels, training_predict))
            plt.show()

    @staticmethod
    def clear_datasets():
        for fname in ("bedroom_dataset.csv", "train.csv", "test.csv"):
            df = pd.read_csv(fname)
            df = df[["No1", "No2", "AreSimilar"]]
            df.to_csv(fname, index=False)
