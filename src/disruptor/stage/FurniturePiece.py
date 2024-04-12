import os.path

from disruptor.stage import Room
from disruptor.tools import get_filename_without_extension, create_directory_if_not_exists
from vedo import *


class FurniturePiece:
    def __init__(self, model_path, wall_projection_model_path, floor_projection_model_path, angles_offset):
        (
            self.model_path,
            self.wall_projection_model_path,
            self.floor_projection_model_path,
            self.angles_offset
        ) = (
            model_path,
            wall_projection_model_path,
            floor_projection_model_path,
            angles_offset
        )

    def render_model(self, render_directory, angles: tuple[int, int, int]):
        for obj_path in (self.model_path, self.wall_projection_model_path, self.floor_projection_model_path):
            vp = Plotter(axes=0, offscreen=True)
            mesh = load(obj_path)
            # mesh = load("3ds/bed.obj").texture("Texture/20430_cat_diff_v1.jpg")
            # mesh.lighting('glossy') # change lighting (press k interactively)
            angle_x, angle_y, angle_z = angles
            offset_x, offset_y, offset_z = self.angles_offset
            mesh.rotateX(angle_x + offset_x)
            mesh.rotateY(angle_y + offset_y)
            mesh.rotateZ(angle_z + offset_z)
            vp += mesh
            vp.show()
            print("showed")
            create_directory_if_not_exists(render_directory)
            save_path = os.path.join(render_directory, get_filename_without_extension(obj_path) + '.png')
            screenshot(save_path)


class Bed(FurniturePiece):
    def __init__(self, model_path='disruptor/stage/3Ds/bedroom/bed/bed.obj',
                 wall_projection_model_path='disruptor/stage/3Ds/bedroom/bed/bed_back.obj',
                 floor_projection_model_path='disruptor/stage/3Ds/bedroom/bed/bed_bottom.obj',
                 angles_offset=(0, 90, 0)):
        super().__init__(model_path, wall_projection_model_path, floor_projection_model_path, angles_offset)
