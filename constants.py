from enum import Enum
import os


def join(directory, file):
    return os.path.join(directory, file)


class Path(Enum):
    # /image
    # pasiba Arsenu 😘😘😘
    IMAGES_DIR = os.path.abspath('images')
    INPUT_IMAGE = join(IMAGES_DIR, 'user_image.png')
    OUTPUT_IMAGE = join(IMAGES_DIR, 'applied.jpg')

    # /image/preprocessed
    PREPROCESSED_IMAGES_DIR = join(IMAGES_DIR, 'preprocessed')
    DESIGNED_IMAGE = join(PREPROCESSED_IMAGES_DIR, 'designed.png')
    FLOOR_LAYOUT_IMAGE = join(PREPROCESSED_IMAGES_DIR, 'floor_layout.png')
    FLOOR_MASK_IMAGE = join(PREPROCESSED_IMAGES_DIR, 'floor_mask.png')
    FURNITURE_MASK_IMAGE = join(PREPROCESSED_IMAGES_DIR, 'furniture_mask.png')
    FURNITURE_PIECE_MASK_IMAGE = join(PREPROCESSED_IMAGES_DIR, 'furniture_piece_mask.png')
    INPAINTING_MASK_IMAGE = join(PREPROCESSED_IMAGES_DIR, 'inpainting_mask.png')
    PREREQUISITE_IMAGE = join(PREPROCESSED_IMAGES_DIR, 'prerequisite.png')
    SEG_PREREQUISITE_IMAGE = join(PREPROCESSED_IMAGES_DIR, 'seg_prerequisite.png')
    SEGMENTED_ES_IMAGE = join(PREPROCESSED_IMAGES_DIR, 'segmented_es.png')
    STRETCHED_WINDOWS_MASK_INPAINTING = join(PREPROCESSED_IMAGES_DIR, 'stretched_windows_mask_inpainting.png')
    WALL_MASK_IMAGE = join(PREPROCESSED_IMAGES_DIR, 'wall_mask.png')
    WINDOWS_MASK_IMAGE = join(PREPROCESSED_IMAGES_DIR, 'windows_mask.png')
    WINDOWS_MASK_INPAINTING_IMAGE = join(PREPROCESSED_IMAGES_DIR, 'windows_mask_inpainting.png')
    PREPROCESSED_USERS = join(PREPROCESSED_IMAGES_DIR,  'users.png')


    # /3Ds
    MODELS_DIR = '3Ds'

    # /3Ds/other
    OTHER_MODELS_DIR = join(MODELS_DIR, 'other')
    CURTAIN_MODEL = join(OTHER_MODELS_DIR, 'curtain.usdc')
    PLANT_MODEL = join(OTHER_MODELS_DIR, 'plant.usdc')

    # /3Ds/living_room
    LIVING_ROOM_MODELS_DIR = join(MODELS_DIR, 'living_room')
    SOFA_WITH_TABLE_MODEL = join(LIVING_ROOM_MODELS_DIR, 'sofa_with_table.usdc')

    # /3Ds/kitchen
    KITCHEN_MODELS_DIR = join(MODELS_DIR, 'kitchen')
    KITCHEN_TABLE_WITH_CHAIRS_MODEL = join(KITCHEN_MODELS_DIR, 'kitchen_table_with_chairs.usdc')

    # /3Ds/bedroom
    BEDROOM_MODELS_DIR = join(MODELS_DIR, 'bedroom')
    BED_MODEL = join(BEDROOM_MODELS_DIR, 'bed.usdc')

    # DepthAnything
    DEPTH_IMAGE = 'DepthAnythingV2/output/depth.npy'
    PLY_SPACE = 'DepthAnythingV2/output/depth.ply'
    DEPTH_CHECKPOINT = 'DepthAnythingV2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth'
    IMAGE_HEIGHT_LIMIT = 512 # To avoid GPU OOM error
    FLOOR_NPY = 'DepthAnythingV2/output/floor.npy'
    FLOOR_PLY = 'DepthAnythingV2/output/floor.ply'

    # UprightNet
