from enum import Enum
import os


def join(directory, file):
    return os.path.join(directory, file)


class Config(Enum):
    IMAGE_HEIGHT_LIMIT = 512  # To avoid GPU OOM error
    UI = 'webui'  # Or 'comfyui'
    DO_POSTPROCESSING = False
    CONTROLNET_HEIGHT_LIMIT = 1024


class Path(Enum):
    # pasiba Arsenu 😘😘😘
    # visuals
    VISUALS_DIR = os.path.abspath('visuals')

    # visuals/image
    IMAGES_DIR = join(VISUALS_DIR, 'images')
    INPUT_IMAGE = join(IMAGES_DIR, 'user_image.png')
    OUTPUT_IMAGE = join(IMAGES_DIR, 'applied.jpg')

    # visuals/image/preprocessed
    PREPROCESSED_DIR = join(IMAGES_DIR, 'preprocessed')
    SEG_INPUT_IMAGE = join(PREPROCESSED_DIR, 'segmented_es.png')
    SEG_RENDER_IMAGE = join(PREPROCESSED_DIR, 'seg_prerequisite.png')
    RENDER_IMAGE = join(PREPROCESSED_DIR, 'prerequisite.png')
    FLOOR_POINTS_IMAGE = join(PREPROCESSED_DIR, 'floor_points.png')
    FLOOR_LAYOUT_IMAGE = join(PREPROCESSED_DIR, 'floor_layout.png')
    FLOOR_LAYOUT_DEBUG_IMAGE = join(PREPROCESSED_DIR, 'floor_layout_debug.png')
    FLOOR_MASK_IMAGE = join(PREPROCESSED_DIR, 'floor_mask.png')
    WINDOWS_MASK_IMAGE = join(PREPROCESSED_DIR, 'windows_mask.png')
    WINDOWS_MASK_INPAINTING_IMAGE = join(PREPROCESSED_DIR, 'windows_mask_inpainting.png')
    STRETCHED_WINDOWS_MASK_INPAINTING_IMAGE = join(PREPROCESSED_DIR, 'stretched_windows_mask_inpainting.png')
    DESIGNED_IMAGE = join(PREPROCESSED_DIR, 'designed.png')

    # visuals/3Ds
    MODELS_DIR = join(VISUALS_DIR, '3Ds')
    SCENE_FILE = join(MODELS_DIR, 'scene.blend')

    # visuals/3Ds/other
    OTHER_MODELS_DIR = join(MODELS_DIR, 'other')
    CURTAIN_MODEL = join(OTHER_MODELS_DIR, 'curtain.usdc')
    PLANT_MODEL = join(OTHER_MODELS_DIR, 'plant.usdc')

    # visuals/3Ds/living_room
    LIVING_ROOM_MODELS_DIR = join(MODELS_DIR, 'living_room')
    SOFA_WITH_TABLE_MODEL = join(LIVING_ROOM_MODELS_DIR, 'sofa_with_table.usdc')

    # visuals/3Ds/kitchen
    KITCHEN_MODELS_DIR = join(MODELS_DIR, 'kitchen')
    KITCHEN_TABLE_WITH_CHAIRS_MODEL = join(KITCHEN_MODELS_DIR, 'kitchen_table_with_chairs.usdc')
    KITCHEN_TABLE_WITH_CHAIRS_MODEL2 = join(KITCHEN_MODELS_DIR, 'moreChairs.usdc')
    KITCHEN_SET_MODEL = join(KITCHEN_MODELS_DIR, 'Alta_2_0-only.usdc')
    KITCHEN_SET_MODEL2 = join(KITCHEN_MODELS_DIR, 'kitchenSetTest.usdc')


    # visuals/3Ds/bedroom
    BEDROOM_MODELS_DIR = join(MODELS_DIR, 'bedroom')
    BED_MODEL = join(BEDROOM_MODELS_DIR, 'bed2.usdc')
    BED_WITH_TABLES_MODEL = join(BEDROOM_MODELS_DIR, 'bedwithtables.usdc')
    WARDROBE_MODEL = join(BEDROOM_MODELS_DIR, 'Madrid_Shafa-3-V2__whithe.usdc')
    COMMODE_MODEL = join(BEDROOM_MODELS_DIR, 'commode2.usdc')
    PAINTING_MODEL = join(BEDROOM_MODELS_DIR, 'painting.usdc')

    # We run it with subprocess to reset all the context for Blender after each scene render
    BLENDER_SCRIPT = os.path.abspath('blender_script.py')

    # DepthAnything
    DEPTH_ANYTHING_DIR = os.path.abspath('DepthAnythingV2')
    DEPTH_IMAGE = join(DEPTH_ANYTHING_DIR, 'output/depth.npy')
    PLY_SPACE = join(DEPTH_ANYTHING_DIR, 'output/depth.ply')
    DEPTH_CHECKPOINT = join(DEPTH_ANYTHING_DIR, 'metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    FLOOR_NPY = join(DEPTH_ANYTHING_DIR, 'output/floor.npy')
    FLOOR_PLY = join(DEPTH_ANYTHING_DIR, 'output/floor.ply')
