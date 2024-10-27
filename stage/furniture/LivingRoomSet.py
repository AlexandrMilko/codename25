from .Furniture import FloorFurniture
from constants import Path
import numpy as np
import cv2


class LivingRoomSet(FloorFurniture):
    def __init__(self, model_path=Path.LIVING_ROOM_SET.value):
        super().__init__(model_path)