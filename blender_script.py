import json
import math
import sys
import bpy


def clean_scene():
    if bpy.context.active_object and bpy.context.active_object.mode == "EDIT":
        bpy.ops.object.editmode_toggle()

    # make sure none of the objects are hidden from the viewport, selection, or disabled
    for obj in bpy.data.objects:
        obj.hide_set(False)
        obj.hide_select = False
        obj.hide_viewport = False

        # select all the object and delete them
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # find all the collections and remove them
    collection_names = [col.name for col in bpy.data.collections]
    for name in collection_names:
        bpy.data.collections.remove(bpy.data.collections[name])

    # in the case when you modify the world shader
    # delete and recreate the world object
    world_names = [world.name for world in bpy.data.worlds]
    for name in world_names:
        bpy.data.worlds.remove(bpy.data.worlds[name])
    # create a new world data block
    bpy.ops.world.new()
    bpy.context.scene.world = bpy.data.worlds["World"]

    # Remove all orphan data blocks
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


def create_node(node_tree, type_name, node_x_location, node_location_step_x=0):
    # Create and position a new node in the node tree
    node_obj = node_tree.nodes.new(type=type_name)
    node_obj.location.x = node_x_location
    node_x_location += node_location_step_x
    return node_obj, node_x_location


def create_material(name="Material"):
    # Create a new material and enable nodes
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    material.node_tree.nodes.new(type="ShaderNodeAttribute")

    # Set attribute node and link to BSDF base color
    principled_bsdf_node = material.node_tree.nodes["Principled BSDF"]
    attribute_node = material.node_tree.nodes["Attribute"]
    attribute_node.attribute_name = "Col"
    material.node_tree.links.new(attribute_node.outputs["Color"], principled_bsdf_node.inputs['Base Color'])

    return material


def create_geo_node_tree():
    node_tree = bpy.data.node_groups["Geometry Nodes"]

    # Add and link geometry nodes for mesh and material processing
    in_node = node_tree.nodes["Group Input"]
    out_node = node_tree.nodes["Group Output"]

    node_x_location = -175
    node_location_step_x = 175

    # Set Material node
    set_material_node, node_x_location = create_node(node_tree, "GeometryNodeSetMaterial",
                                                     node_x_location, node_location_step_x)
    set_material_node.inputs[2].default_value = bpy.data.materials["Material"]
    node_tree.links.new(in_node.outputs["Geometry"], set_material_node.inputs['Geometry'])
    node_tree.links.new(set_material_node.outputs["Geometry"], out_node.inputs['Geometry'])


def import_room(path):
    # Import a room model and apply geometry node setup
    bpy.ops.wm.ply_import(filepath=path)
    bpy.ops.node.new_geometry_nodes_modifier()

    create_material()
    create_geo_node_tree()
    bpy.context.object.visible_shadow = False


def setup_camera(angles, location, focal_length_px, image_width_px):
    # Get the default sensor width in mm (usually 36 mm for a full-frame camera in Blender)
    sensor_width_mm = bpy.data.cameras.new(name='Camera').sensor_width

    # Convert focal length from pixels to millimeters
    focal_length_mm = (focal_length_px * sensor_width_mm) / image_width_px

    # Create a new camera object
    cam_data = bpy.data.cameras.new(name='Camera')
    cam_obj = bpy.data.objects.new('Camera', cam_data)

    # Link the camera to the scene
    bpy.context.collection.objects.link(cam_obj)

    # Set the camera's location, rotation, and focal length in mm
    cam_obj.location = location
    cam_obj.rotation_euler = angles
    cam_obj.data.lens = focal_length_mm
    bpy.context.scene.camera = cam_obj


def setup_light(has_area_light):
    # Create light data and object
    light_data = bpy.data.lights.new(name='CameraAreaLight', type='AREA')
    light_obj = bpy.data.objects.new(name='CameraAreaLight', object_data=light_data)

    # Link light object to the scene
    bpy.context.collection.objects.link(light_obj)

    # Set light location, intensity, and color
    light_obj.location = (0, 0, 0)
    light_obj.rotation_euler = (math.radians(90), 0, 0)
    light_data.energy = 75 if has_area_light else 130
    light_data.size = 4
    light_data.color = (1, 1, 1)


def add_area_light(light_params):
    """
    Adds two AREA lights based on calculated window parameters.
    """
    # First light
    light_data_1 = bpy.data.lights.new(name='WindowAreaLight1', type='AREA')
    light_obj_1 = bpy.data.objects.new(name='WindowAreaLight1', object_data=light_data_1)
    light_obj_1.location = light_params.get('offset')
    yaw_angle = light_params.get('yaw_angle', 0)
    light_obj_1.rotation_euler = (math.radians(-90), 0, math.radians(yaw_angle))
    light_obj_1.data.shape = 'RECTANGLE'
    light_obj_1.data.size = light_params.get('size', 1.0)/1.5  # Width of the light
    light_obj_1.data.size_y = light_params.get('size_y', 1.0)/1.5  # Height of the light
    light_obj_1.data.energy = light_params.get('energy', 70.0)
    light_obj_1.data.color = light_params.get('color', (1.0, 1.0, 1.0))
    light_obj_1.data.shadow_soft_size = light_params.get('shadow_soft_size', 1.0)
    bpy.context.collection.objects.link(light_obj_1)
    print(f"Light size (Width x Height) = {light_obj_1.data.size} x {light_obj_1.data.size_y}")
    #
    # # Second light (rotated by -90 degrees on X-axis)
    light_data_2 = bpy.data.lights.new(name='WindowAreaLi ght2', type='AREA')
    light_obj_2 = bpy.data.objects.new(name='WindowAreaLight2', object_data=light_data_2)
    light_obj_2.location = light_params.get('offset')
    light_obj_2.rotation_euler = (math.radians(90), 0, math.radians(yaw_angle))  # Adjusted rotation on X-axis
    light_obj_2.data.shape = 'RECTANGLE'
    light_obj_2.data.size = light_params.get('size', 1.0)/1.5  # Width of the light
    light_obj_2.data.size_y = light_params.get('size_y', 1.0)/1.5  # Height of the light
    light_obj_2.data.energy = light_params.get('energy', 5.0)
    light_obj_2.data.color = light_params.get('color', (1.0, 1.0, 1.0))
    light_obj_2.data.shadow_soft_size = light_params.get('shadow_soft_size', 1.0)
    bpy.context.collection.objects.link(light_obj_2)
    print(f"Second light added with size (Width x Height) = {light_obj_2.data.size} x {light_obj_2.data.size_y}")


def add_furniture(path, location, angles, scale):
    # Import the 3D model
    bpy.ops.wm.usd_import(filepath=path)

    # Get the imported objects
    imported_objs = bpy.context.selected_objects

    # Create and link an empty parent object
    furniture = bpy.data.objects.new("Furniture", None)
    bpy.context.collection.objects.link(furniture)

    # Parent imported objects to the empty object
    for obj in imported_objs:
        obj.parent = furniture

    # Set location, rotation, and scale
    furniture.location = location
    furniture.rotation_euler = angles
    furniture.scale = scale


def use_gpu():
    bpy.context.scene.render.engine = 'CYCLES'
    import torch
    if torch.cuda.is_available():
        print("INFO: CUDA is available. Using it for render")
        bpy.context.preferences.addons[
            "cycles"
        ].preferences.compute_device_type = "CUDA"  # or "OPENCL"
    else:
        print("WARNING: CUDA is not available for render.")

    # Set the device and feature set
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.denoising_use_gpu = True

    # get_devices() to let Blender detects GPU device
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1  # Using all devices, include GPU and CPU
        print(d["name"], d["use"])


def set_rendering_parameters(samples):
    # Set up Cycles rendering with GPU, adaptive sampling, and denoising
    bpy.context.scene.render.engine = 'CYCLES'
    use_gpu()
    bpy.context.scene.cycles.samples = samples

    # Adaptive sampling to reduce render times in low-noise areas
    bpy.context.scene.cycles.use_adaptive_sampling = True
    bpy.context.scene.cycles.adaptive_threshold = 0.1  # Noise threshold

    # Enable AI denoising with high-quality settings
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    bpy.context.scene.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
    bpy.context.scene.cycles.denoising_prefilter = 'ACCURATE'


def save_render(path, res_x, res_y, samples):
    set_rendering_parameters(samples)

    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.filepath = path
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.film_transparent = True

    # Render the scene
    bpy.ops.render.render(write_still=True)


def save_blend_file(path):
    bpy.ops.wm.save_as_mainfile(filepath=path)


if __name__ == "__main__":
    args = sys.argv
    data = json.loads(args[1])

    render_path = data['render_path']
    blend_file_path = data['blend_file_path']
    render_samples = data['render_samples']
    room_point_cloud_path = data['room_point_cloud_path']
    focal_length_px = data['focal_length_px']
    camera_location = data['camera_location']
    camera_angles = data['camera_angles']
    resolution_x = data['resolution_x']
    resolution_y = data['resolution_y']
    objects = data['objects']
    lights = data.get('lights', [])

    clean_scene()

    scene = bpy.context.scene
    import_room(room_point_cloud_path)
    setup_camera(camera_angles, camera_location, focal_length_px, resolution_x)

    # Add lights from provided data
    for light_params in lights:
        add_area_light(light_params)
    # Pass bool(lights) to setup_light to adjust energy based on area lights
    setup_light(has_area_light=bool(lights))

    # Add objects/furniture from provided data
    for obj in objects:
        add_furniture(obj['obj_path'], obj['obj_offsets'], obj['obj_angles'], obj['obj_scale'])

    save_render(render_path, resolution_x, resolution_y, render_samples)
    save_blend_file(blend_file_path)
