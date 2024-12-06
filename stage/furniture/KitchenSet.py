from constants import Path

from .Furniture import FloorFurniture


class KitchenSet(FloorFurniture):
    def __init__(self, model_path):
        super().__init__(model_path)
