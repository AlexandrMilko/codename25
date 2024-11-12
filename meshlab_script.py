import sys
import pymeshlab


def screened_poisson_surface_reconstruction():
    mesh_path = sys.argv[1]

    # Create a MeshSet object
    ms = pymeshlab.MeshSet()

    # Import the point cloud file in ".ply" format
    ms.load_new_mesh(mesh_path)

    # Compute normals for point sets
    ms.compute_normal_for_point_clouds()

    # Perform Screened Poisson Surface Reconstruction
    ms.generate_surface_reconstruction_screened_poisson(depth=8, scale=1, samplespernode=2)

    # Select faces with edges longer than a default threshold
    ms.compute_selection_by_edge_length()

    # Remove selected faces
    ms.meshing_remove_selected_faces()

    # Remove isolated pieces with the option to remove unreferenced vertices
    ms.meshing_remove_connected_component_by_face_number()

    # # Perform Per Vertex Texture Function
    # ms.compute_texcoord_by_function_per_vertex()
    #
    # # Convert PerVertex UV into PerWedge UV
    # ms.compute_texcoord_transfer_vertex_to_wedge()
    #
    # # Parametrization: trivial per-triangle with specified parameters
    # ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=4096, border=0, method=0)

    # Export the mesh as a .ply file
    ms.save_current_mesh(mesh_path,
                         save_vertex_quality=False,
                         save_face_quality=False,
                         save_face_color=False,
                         save_wedge_color=False,
                         save_wedge_normal=False)

    print("Mesh processing completed and saved as 'output_mesh.ply'")


if __name__ == '__main__':
    screened_poisson_surface_reconstruction()
