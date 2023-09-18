import cv2
import numpy as np
import pandas as pd
import os

#TODO change to url_for all directory references
default_input_image = 'disruptor/static/images/current_image.jpg'
default_segmented_image = 'disruptor/static/images/preprocessed/preprocessed.jpg'
save_dir = 'disruptor/static/images/parsed_furniture'
def parse_objects(input_image=default_input_image,
                  segmented_image=default_segmented_image):
    # Load the image
    image = cv2.imread(segmented_image)

    # Load the color coding information from the CSV file
    classes_filename = "color_coding_semantic_segmentation_classes.csv"
    classes_path = 'disruptor/static/assets/' + classes_filename
    color_coding_df = pd.read_csv(classes_path)

    # Loop through each row in the CSV file
    for index, row in color_coding_df.iterrows():
        color_code = row['Color_Code']
        color_name = row['Name']

        # Extract the RGB values from the color code
        rgb_values = tuple(map(int, color_code.strip('()').split(',')))[::-1]  # For BGR format

        # Define the lower and upper bounds for the color
        tolerance = 3
        lower_color = np.array([x - tolerance for x in rgb_values])
        upper_color = np.array([x + tolerance for x in rgb_values])

        # Create a mask for the color
        color_mask = cv2.inRange(image, lower_color, upper_color)

        # Create a black and white mask
        bw_mask = np.zeros_like(color_mask)
        bw_mask[color_mask != 0] = 255

        # Check if the mask contains white pixels
        if np.any(bw_mask == 255):
            # Save the resulting mask in the "Objects" directory with the color name as the filename
            mask_filename = os.path.join(save_dir, f'{color_name}.jpg')
            cv2.imwrite(mask_filename, bw_mask)

def unite_groups(input_dir, output_dir, groups):
    # Iterate through each group of masks
    for group in groups:
        # Initialize an empty mask for the group
        group_mask = None

        # Iterate through masks in the group
        for object_name in group:
            # Construct the file path for the mask
            mask_file = os.path.join(input_dir, f"{object_name}.jpg")

            # Check if the mask file exists
            if not os.path.isfile(mask_file):
                print(f"Mask file '{object_name}.jpg' not found.")
                continue

            # Load the mask as a grayscale image
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

            # If group_mask is None, initialize it with the first mask
            if group_mask is None:
                group_mask = mask
            else:
                # Unite the current mask with the group_mask using bitwise OR
                group_mask = cv2.bitwise_or(group_mask, mask)

        # Check if group_mask is empty before saving
        if group_mask is not None and not np.all(group_mask == 0):
            # Save the united mask for the group
            output_file = os.path.join(output_dir, "_".join(group) + ".jpg")
            cv2.imwrite(output_file, group_mask)
        else:
            print(f"No valid masks found for group: {group}")