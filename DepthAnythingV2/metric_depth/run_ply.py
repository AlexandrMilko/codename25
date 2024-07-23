import open3d as o3d
import numpy as np


def transform_to_blender_xyz(x, y, z):  # TODO test it and visualize the whole depth estimation
    # 1. Invert the y
    # 2. Swap the z and y
    # 3. Invert x
    return -x, z, -y


filename = "img"
point_cloud = o3d.io.read_point_cloud(f"output/{filename + '.ply'}")
points = np.asarray(point_cloud.points)
points[:, 0] = -points[:, 0]
points[:, 1] = -points[:, 1]
point_cloud.points = o3d.utility.Vector3dVector(points)

#coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
#o3d.visualization.draw_geometries([point_cloud, point_cloud_main, coordinate_frame])

o3d.visualization.draw_geometries([point_cloud])
