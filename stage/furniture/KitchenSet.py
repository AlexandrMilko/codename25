from constants import Path

from .Furniture import FloorFurniture


class KitchenSet(FloorFurniture):
    def __init__(self, model_path=Path.KITCHEN_SET_MODEL.value):
        # Инициализация родительского класса с моделью
        super().__init__(model_path)