from constants import Path

from .Furniture import FloorFurniture


class KitchenSet(FloorFurniture):
    def __init__(self, model_path=Path.
                 cdKITCHEN_SET_MODEL3.value):
        # Инициализация родительского класса с моделью
        super().__init__(model_path)