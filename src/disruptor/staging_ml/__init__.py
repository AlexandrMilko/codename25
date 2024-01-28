import pandas as pd
import os
from disruptor.green_screen.find_similar.ssim import calculate_vanishing_point_by_XiaohuLu, find_object_centers
from disruptor import app
from flask_login import current_user
from flask import url_for


# TODO Use it for the ML input, with Pipelines too.
# from sklearn.base import BaseEstimator, TransformerMixin
#
# class AttributesAdder(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self
#     def transform(self, X, y=None):
#         pass

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
            self.window_centers = find_object_centers(self.segmented_before, Room.window_color, debug=True)
            self.door_centers = find_object_centers(self.segmented_before, Room.door_color, debug=True)
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
            self.window_centers = find_object_centers(self.tmp_segmented_before, Room.window_color, debug=True)
            self.door_centers = find_object_centers(self.tmp_segmented_before, Room.door_color, debug=True)

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
            re.search(r'\d+', os.path.basename(image_path)).group())
        trio["segmented_before"] = os.path.join(room_directory, "es_segmented/" + image_number + "Before.jpg")
        trio["before"] = os.path.join(room_directory, "original/" + image_number + "Before.jpg")
        trio["after"] = os.path.join(room_directory, "original/" + image_number + "After.jpg")

        return trio

    @property
    def windows_number(self):
        return len(self.window_centers)

    @property
    def doors_number(self):
        return len(self.door_centers)

    def get_features_values(self):
        # We sort them so that the furthest objects are compared with the furthest and the closest with the closest
        features = [self.vanishing_point, *sorted(self.window_centers), *sorted(self.door_centers)]
        return features

    def measure_similarity(self, other):
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        if (self.windows_number == other.windows_number) and (self.doors_number == other.doors_number):
            print(self.get_features_values())
            print(other.get_features_values())
            print()
            features_first = np.array(self.get_features_values())
            features_second = np.array(other.get_features_values())
            # print(features_first)
            # print(features_second)
            # print()
            first_flattened = np.reshape(features_first, (1, -1))
            second_flattened = np.reshape(features_second, (1, -1))
            # print(first_flattened)
            # print(second_flattened)
            # print()
            similarity_matrix = cosine_similarity(first_flattened, second_flattened)
            return similarity_matrix[0][0]
        else:
            return 0