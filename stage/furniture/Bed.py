from .Furniture import FloorFurniture
from constants import Path

class Bed(FloorFurniture):
    def __init__(self, model_path=Path.BED_MODEL.value):
        super().__init__(model_path)