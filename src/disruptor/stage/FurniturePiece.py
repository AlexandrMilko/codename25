import os.path

from disruptor.stage import Room
from disruptor.tools import get_filename_without_extension, create_directory_if_not_exists
# from vedo import *


class FurniturePiece:
    scale = 1, 1, 1
    offset_angles = 0, 0, 0

    def __init__(self, model_path, wall_projection_model_path, floor_projection_model_path):
        (
            self.model_path,
            self.wall_projection_model_path,
            self.floor_projection_model_path,
        ) = (
            model_path,
            wall_projection_model_path,
            floor_projection_model_path,
        )

#    def render_model(self, render_directory, angles: tuple[int, int, int]):
#        for obj_path in (self.model_path, self.wall_projection_model_path, self.floor_projection_model_path):
#            vp = Plotter(axes=0, offscreen=True)
#            mesh = load(obj_path)
#            # mesh = load("3ds/bed.obj").texture("Texture/20430_cat_diff_v1.jpg")
#            # mesh.lighting('glossy') # change lighting (press k interactively)
#            angle_x, angle_y, angle_z = angles
#            mesh.rotateX(angle_x)
#            mesh.rotateY(angle_y)
#            mesh.rotateZ(angle_z)
#            vp += mesh
#            vp.show()
#            print("showed")
#            create_directory_if_not_exists(render_directory)
#            save_path = os.path.join(render_directory, get_filename_without_extension(obj_path) + '.png')
#            screenshot(save_path)

    def get_scale(self):
        return self.scale

    def get_offset_angles(self):
        return self.offset_angles


class Bed(FurniturePiece):
    # We use it to scale the model to metric units
    scale = 0.01, 0.01, 0.01
    # We use it to compensate the angle if the model is originally rotated in a wrong way
    offset_angles = 0, 0, 90

    def __init__(self, model_path='disruptor/stage/3Ds/bedroom/bed/bed.obj',
                 wall_projection_model_path='disruptor/stage/3Ds/bedroom/bed/bed_back.obj',
                 floor_projection_model_path='disruptor/stage/3Ds/bedroom/bed/bed_bottom.obj',
                 ):
        super().__init__(model_path, wall_projection_model_path, floor_projection_model_path)
