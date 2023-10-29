from PIL import Image
import os

# Specify the input directory where your images are located
input_directory = 'good_pics/verticalVSjpg'

# Specify the output directory where you want to save the split images
output_directory = 'good_pics/splithorizontal'

number_of_images_we_have_already = 128

# Get a list of image files in the input directory with .jpg and .png extensions
image_files = [f for f in os.listdir(input_directory) if f.endswith(('.jpg', '.png'))]

# Loop through each image file
for index, image_file in enumerate(image_files):
    # Open the image
    image = Image.open(os.path.join(input_directory, image_file))

    # Get the dimensions of the image
    width, height = image.size

    # Calculate the position to split the image in half horizontally
    split_point = height // 2

    # Crop the left and right halves
    left_half = image.crop((0, 0, width, split_point))
    right_half = image.crop((0, split_point, width, height))

    # Determine the file extension (jpg or png) for saving
    file_extension = os.path.splitext(image_file)[1].lower()

    # Save the left and right halves in the output directory with appropriate filenames
    left_half.save(os.path.join(output_directory, f'{1 + index + number_of_images_we_have_already}Before{file_extension}'))
    right_half.save(os.path.join(output_directory, f'{1 + index + number_of_images_we_have_already}After{file_extension}'))

    print(f'Split and saved {image_file} into {1 + index + number_of_images_we_have_already}Before{file_extension} and {1 + index + number_of_images_we_have_already}After{file_extension}')

print('Splitting and saving complete.')
