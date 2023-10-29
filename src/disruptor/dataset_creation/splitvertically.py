from PIL import Image
import os

# Specify the input directory where your images are located
input_directory = 'good_pics/horizontalVSjpg'

# Specify the output directory where you want to save the split images
output_directory = 'good_pics/splitvertical'

number_of_images_we_have_already = 491

# Get a list of image files in the input directory with .jpg and .png extensions
image_files = [f for f in os.listdir(input_directory) if f.endswith(('.jpg', '.png'))]

# Loop through each image file
for index, image_file in enumerate(image_files):
    # Open the image
    image = Image.open(os.path.join(input_directory, image_file))

    # Get the dimensions of the image
    width, height = image.size

    # Calculate the position to split the image in half horizontally
    split_point = width // 2

    # Crop the left and right halves
    top_half = image.crop((0, 0, split_point, height))
    bottom_half = image.crop((split_point, 0, width, height))

    # Determine the file extension (jpg or png) for saving
    file_extension = os.path.splitext(image_file)[1].lower()

    # Save the left and right halves in the output directory with appropriate filenames
    top_half.save(os.path.join(output_directory, f'{1 + index + number_of_images_we_have_already}Before{file_extension}'))
    bottom_half.save(os.path.join(output_directory, f'{1 + index + number_of_images_we_have_already}After{file_extension}'))

    print(f'Split and saved {image_file} into {1 + index + number_of_images_we_have_already}Before{file_extension} and {1 + index + number_of_images_we_have_already}After{file_extension}')

print('Splitting and saving complete.')
