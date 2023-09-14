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