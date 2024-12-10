from constants import Path
# from lang_segment_anything.lang_sam import LangSAM
from PIL import Image
from io import BytesIO
from lang_segment_anything.lang_sam.utils import draw_image
import numpy as np
import base64

from lang_segment_anything.lang_sam.lang_sam import LangSAM

SAM_TYPE = "sam2.1_hiera_small"
model = LangSAM(sam_type=SAM_TYPE)

def predict(inputs: dict) -> dict:
    """Perform prediction using the LangSAM model.

    Yields:
        dict: Contains the processed output image.
    """
    print("Starting prediction with parameters:")
    print(
        f"sam_type: {inputs['sam_type']}, \
            box_threshold: {inputs['box_threshold']}, \
            text_threshold: {inputs['text_threshold']}, \
            text_prompt: {inputs['text_prompt']}"
    )

    if inputs["sam_type"] != model.sam_type:
        print(f"Updating SAM model type to {inputs['sam_type']}")
        model.sam.build_model(inputs["sam_type"])

    try:
        image_pil = Image.open(BytesIO(inputs["image_bytes"])).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")

    results = model.predict(
        images_pil=[image_pil],
        texts_prompt=[inputs["text_prompt"]],
        box_threshold=inputs["box_threshold"],
        text_threshold=inputs["text_threshold"],
    )
    results = results[0]

    if not len(results["masks"]):
        print("No masks detected. Returning original image.")
        return {"output_image": image_pil}

    # Draw results on the image
    image_array = np.asarray(image_pil)
    output_image = draw_image(
        image_array,
        results["masks"],
        results["boxes"],
        results["scores"],
        results["labels"],
    )
    output_image = Image.fromarray(np.uint8(output_image)).convert("RGB")
    # Convert bboxes to integers
    results["boxes"].flags.writeable = False  # Simulate a non-writable array
    # Create a writable copy and convert values to integers
    bboxes = results["boxes"].copy().astype(int)

    return {"output_image": output_image, "boxes": bboxes}




# if __name__ == "__main__":
#     inputs = {
#         "sam_type": SAM_TYPE,
#         "box_threshold": 0.3,
#         "text_threshold": 0.25,
#         "text_prompt": "doorway",
#         "image_bytes": get_image_bytes(Path.INPUT_IMAGE.value),
#     }
#     output = predict(inputs)
#     output_image = output[Path.DOOR_SEG_IMG_OUTPUT.value]
#     output_image.save(Path.DOOR_SEG_IMG_OUTPUT.value, format="PNG")