from .Furniture import FloorFurniture
from constants import Path


class Commode(FloorFurniture):
    def __init__(self, model_path=Path.COMMODE_MODEL.value):
        super().__init__(model_path)
