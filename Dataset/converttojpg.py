from PIL import Image
import os
import shutil

def convert_and_copy_images(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through files in the input directory
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            # If the file is JPG or PNG, just copy it to the output directory
            shutil.copy(input_path, output_path)
            print(f"Copied: {input_path} to {output_path}")
        elif filename.lower().endswith(".webp"):
            try:
                # Open the WebP image and convert it to JPG
                with Image.open(input_path) as img:
                    output_path = output_path.split(".")[0] + ".jpg"
                    img.convert("RGB").save(output_path, "JPEG")
                    print(f"Converted: {input_path} to {output_path}")
            except Exception as e:
                print(f"Error converting {input_path}: {e}")

if __name__ == "__main__":
    input_directory = "good_pics/VerticalVS"  # Replace with the directory containing images
    output_directory = "good_pics/verticalVSjpg"  # Replace with the directory where you want to save the images

    convert_and_copy_images(input_directory, output_directory)
