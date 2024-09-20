import open3d as o3d
import numpy as np

filename = "img"
point_cloud = o3d.io.read_point_cloud(f"output/{filename + '.ply'}")
points = np.asarray(point_cloud.points)
point_cloud.points = o3d.utility.Vector3dVector(points)

#coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
#o3d.visualization.draw_geometries([point_cloud, point_cloud_main, coordinate_frame])

o3d.visualization.draw_geometries([point_cloud])
