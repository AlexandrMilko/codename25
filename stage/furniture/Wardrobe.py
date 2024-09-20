from .Furniture import FloorFurniture
from constants import Path

class Wardrobe(FloorFurniture):
    def __init__(self, model_path=Path.WARDROBE_MODEL.value):
        super().__init__(model_path)