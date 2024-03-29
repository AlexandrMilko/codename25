from vedo import *

def render_model(obj_path, render_path, angles: tuple[int, int, int]):
    vp = Plotter(axes=0, offscreen=False)
    mesh = load(obj_path)
    # mesh = load("3ds/bed.obj").texture("Texture/20430_cat_diff_v1.jpg")
    # mesh.lighting('glossy') # change lighting (press k interactively)
    angle_x, angle_y, angle_z = angles
    mesh.rotate_x(angle_x)
    mesh.rotate_y(angle_y)
    mesh.rotate_z(angle_z)
    vp += mesh
    vp.show()
    screenshot(render_path)