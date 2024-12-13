from enum import Enum
import os


def join(directory, file):
    return os.path.join(directory, file)


class Config(Enum):
    IMAGE_HEIGHT_LIMIT = 1080  # For limiting the time it takes to render and depth calculation for the image
    RENDER_SAMPLES = 64  # Lower samples: faster render times but reduced image quality
    DO_POSTPROCESSING = False
    CONTROLNET_HEIGHT_LIMIT = 1024
    FLOOR_LAYOUT_CONTOUR_SIZE_TO_REMOVE = 1000


class URL(Enum):
    SERVER = 'http://127.0.0.1:8188/'
    WS = 'ws://127.0.0.1:8188/ws?clientId='


class Path(Enum):
    # pasiba Arsenu 😘😘😘
    # visuals
    VISUALS_DIR = os.path.abspath('visuals')

    # visuals/image
    IMAGES_DIR = join(VISUALS_DIR, 'images')
    INPUT_IMAGE = join(IMAGES_DIR, 'user_image.png')

    # visuals/image/preprocessed
    PREPROCESSED_DIR = join(IMAGES_DIR, 'preprocessed')
    SEG_INPUT_IMAGE = join(PREPROCESSED_DIR, 'segmented_es.png')
    SEG_RENDER_IMAGE = join(PREPROCESSED_DIR, 'seg_prerequisite.png')
    RENDER_IMAGE = join(PREPROCESSED_DIR, 'prerequisite.jpg')
    FLOOR_POINTS_IMAGE = join(PREPROCESSED_DIR, 'floor_points.png')
    FLOOR_LAYOUT_IMAGE = join(PREPROCESSED_DIR, 'floor_layout.png')
    FLOOR_LAYOUT_DEBUG_IMAGE = join(PREPROCESSED_DIR, 'floor_layout_debug.png')
    FLOOR_MASK_IMAGE = join(PREPROCESSED_DIR, 'floor_mask.png')
    WINDOWS_MASK_IMAGE = join(PREPROCESSED_DIR, 'windows_mask.png')
    WINDOWS_MASK_INPAINTING_IMAGE = join(PREPROCESSED_DIR, 'windows_mask_inpainting.png')
    STRETCHED_WINDOWS_MASK_INPAINTING_IMAGE = join(PREPROCESSED_DIR, 'stretched_windows_mask_inpainting.png')
    DESIGNED_IMAGE = join(PREPROCESSED_DIR, 'designed.png')
    DEPTH_DEBUG_IMAGE = join(PREPROCESSED_DIR, 'depth_image.png')
    WALL_SEGMENTS_DEBUG_IMAGE = join(PREPROCESSED_DIR, 'last_wall_segment.png')
    DOOR_SEG_IMG_OUTPUT = join(PREPROCESSED_DIR, 'doorway_seg.png')
    REDUNDANT_WALLS_ON_FLOOR_MASK_DEBUG_IMAGE = join(PREPROCESSED_DIR, 'redundant_walls_on_floor_mask.png')

    OUTPUT_IMAGE = join(PREPROCESSED_DIR, 'applied.jpg')

    # visuals/3Ds
    MODELS_DIR = join(VISUALS_DIR, '3Ds')
    SCENE_FILE = join(MODELS_DIR, 'scene.blend')

    # visuals/3Ds/other
    OTHER_MODELS_DIR = join(MODELS_DIR, 'other')
    CURTAIN_MODEL = join(OTHER_MODELS_DIR, 'curtain.usdc')
    PLANT_MODEL = join(OTHER_MODELS_DIR, 'plant.usdc')

    # visuals/3Ds/living_room
    LIVING_ROOM_MODELS_DIR = join(MODELS_DIR, 'living_room')
    LIVING_ROOM_SET = join(LIVING_ROOM_MODELS_DIR, 'living_room_set.usdc')

    # visuals/3Ds/kitchen
    KITCHEN_MODELS_DIR = join(MODELS_DIR, 'kitchen')
    KITCHEN_TABLE_MODEL_ONE = join(KITCHEN_MODELS_DIR, 'tableOne.usdc')
    KITCHEN_TABLE_MODEL_TWO = join(KITCHEN_MODELS_DIR, 'tableTwo.usdc')
    KITCHEN_BIG_MODEL = join(KITCHEN_MODELS_DIR, 'bigOne.usdc')
    KITCHEN_SMALL_ONE = join(KITCHEN_MODELS_DIR, 'smallOne.usdc')
    KITCHEN_SMALL_TWO = join(KITCHEN_MODELS_DIR, 'smallTwo.usdc')
    KITCHEN_SMALL_THREE = join(KITCHEN_MODELS_DIR, 'smallThree.usdc')

    # visuals/3Ds/bedroom
    BEDROOM_MODELS_DIR = join(MODELS_DIR, 'bedroom')
    BED_MODEL = join(BEDROOM_MODELS_DIR, 'bed2.usdc')
    BED_WITH_TABLES_MODEL = join(BEDROOM_MODELS_DIR, 'bedwithtables.usdc')
    WARDROBE_MODEL = join(BEDROOM_MODELS_DIR, 'Madrid_Shafa-3-V2__whithe.usdc')
    COMMODE_MODEL = join(BEDROOM_MODELS_DIR, 'commode2.usdc')
    PAINTING_MODEL = join(BEDROOM_MODELS_DIR, 'painting.usdc')

    # ML_DEPTH_PRO
    ML_DEPTH_PRO_DIR = os.path.abspath('ml_depth_pro')
    ML_DEPTH_PRO_CHECKPOINT = join(ML_DEPTH_PRO_DIR, 'src/depth_pro/cli/checkpoints/depth_pro.pt')
    DEPTH_NPY = join(ML_DEPTH_PRO_DIR, 'output/depth.npy')
    DEPTH_PLY = join(ML_DEPTH_PRO_DIR, 'output/depth.ply')
    FLOOR_NPY = join(ML_DEPTH_PRO_DIR, 'output/floor.npy')
    FLOOR_PLY = join(ML_DEPTH_PRO_DIR, 'output/floor.ply')

    # We run it with subprocess to reset all the context after each scene render
    BLENDER_SCRIPT = os.path.abspath('blender_script.py')
    MESHLAB_SCRIPT = os.path.abspath('meshlab_script.py')
