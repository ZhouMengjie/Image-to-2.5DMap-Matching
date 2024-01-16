import numbers
import open3d as o3d

# Replace 'path/to/your/file.obj' with the actual path to your .obj file
file_path = 'stanford-bunny.obj'

# Load the .obj file
mesh = o3d.io.read_triangle_mesh(file_path)
mesh.paint_uniform_color([1, 0.706, 0])
mesh.compute_vertex_normals()

# Visualize the mesh
# o3d.visualization.draw_geometries([mesh], window_name='Open3D Mesh Visualization')

# Sample and visualize mesh to the point cloud
numbers_points = len(mesh.vertices)*2
pcd = mesh.sample_points_uniformly(number_of_points=numbers_points)
# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name="Open3D PCD Visualization")
# vis.get_render_option().point_size = 2
# opt = vis.get_render_option()
# vis.add_geometry(pcd)
# vis.run()
# vis.destroy_window()

# Sample and visualize mesh to the voxel
# voxel_size = 0.002
# pcd.estimate_normals()
voxel = pcd.voxel_down_sample(voxel_size=0.002)
# voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
# o3d.visualization.draw_geometries([voxel], window_name='Open3D Voxel Visualization')
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Open3D PCD Visualization")
vis.get_render_option().point_size = 2
opt = vis.get_render_option()
vis.add_geometry(voxel)
vis.run()
vis.destroy_window()



