import bpy
from math import radians
import os

def setup_camera(angles, location):
    # Get the current scene
    scene = bpy.context.scene

    # Create a new camera object
    cam_data = bpy.data.cameras.new(name='Camera')
    cam_obj = bpy.data.objects.new('Camera', cam_data)

    # Link the camera to the scene
    scene.collection.objects.link(cam_obj)

    # Set the camera's location
    cam_obj.location = location

    # Set the camera's rotation (in radians)
    cam_obj.rotation_euler = angles

    cam_obj.data.lens = 14
    scene.camera = cam_obj

def setup_light():
    scene = bpy.context.scene
    # Add a point light
    light_data = bpy.data.lights.new(name='PointLight', type='POINT')
    light_obj = bpy.data.objects.new(name='PointLight', object_data=light_data)
    scene.collection.objects.link(light_obj)

    # Set light location
    light_obj.location = (2, 2, 5)

    # Set light energy
    light_data.energy = 1000.0

    # Set light color
    light_data.color = (1, 1, 1)

def add_obj_model(obj_path, location, angles, scale):
    scene = bpy.context.scene
    # Load your .obj model
    bpy.ops.wm.obj_import(filepath=obj_path)
    # Assuming your .obj model is imported as the active object
    # Get a list of all the objects in the scene
    imported_objs = bpy.context.selected_objects
    # Create an Empty object to act as the parent
    furniture = bpy.data.objects.new("Furniture", None)
    scene.collection.objects.link(furniture)

    # Parent all the imported objects to the Empty object
    for obj in imported_objs:
        obj.parent = furniture

    furniture.location = location
    furniture.rotation_euler = angles
    furniture.scale = scale

def save_render(path):
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = path
    
    # Render the scene
    bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    obj_offsets = 0.89264309, 4.07660634, 0.
    obj_angles = 0.0, 0.0, 0.371277267400237
    obj_scale = 0.01, 0.01, 0.01
    camera_angles = 1.6597176418922286, -0.00024912030799756606, 0
    camera_location = 0, 0, 1.3629773865303363
    
    render_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'render.png')
    obj_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'bed_back_centroid.obj')
    setup_camera(camera_angles, camera_location)
    setup_light()
    add_obj_model(obj_path, obj_offsets, obj_angles, obj_scale)
    save_render(render_path)