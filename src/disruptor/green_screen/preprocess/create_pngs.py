from PIL import Image
import os

def create_fg(mask_dir, staged_img_path):
    # Open the main image
    image = Image.open(staged_img_path)

    # Create a new image with an alpha channel using the main image
    result = Image.new("RGBA", image.size)

    # List all mask files in the "masks/" directory
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".jpg")]

    # Composite each mask onto the result image
    for mask_file in mask_files:
        # Open the mask image
        mask = Image.open(os.path.join(mask_dir, mask_file))

        # Ensure both images have the same dimensions
        if mask.size != image.size:
            mask = mask.resize(image.size)

        # Overlay the mask onto the result image
        result.paste(image, (0, 0), mask)

        # Close the mask image
        mask.close()

    # Save the result as a single PNG image
    result.save("disruptor/static/images/preprocessed/foreground.png")

    # Close the main image
    image.close()

def overlay(es_image, fg_image):
    import PIL

    # Open the JPG and PNG images
    jpg_image = PIL.Image.open(es_image)
    png_image = PIL.Image.open(fg_image)

    # Make sure the PNG image has an alpha channel for transparency
    png_image = png_image.convert("RGBA")

    # Resize the PNG image to fit on the JPG image if needed
    png_image = png_image.resize(jpg_image.size, PIL.Image.Resampling.LANCZOS)

    # Overlay the PNG image onto the JPG image
    jpg_image.paste(png_image, (0, 0), png_image)
    jpg_image = jpg_image.convert("RGB")

    # Save the resulting image
    jpg_image.save('disruptor/static/images/preprocessed/prerequisite.jpg')

    # Close the images
    jpg_image.close()
    png_image.close()
