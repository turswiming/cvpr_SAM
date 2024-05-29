import open3d as o3d
import json
import numpy as np


class Wall:
    def __init__(self,
                 id: int,
                 wall: o3d.geometry.AxisAlignedBoundingBox,
                 width: float,
                 height: float, neighbor_wall_ids_at_start: list,
                 neighbor_wall_ids_at_end: list):
        self.id: int = id
        self.wall: o3d.geometry.AxisAlignedBoundingBox = wall
        self.width: float = width
        self.height: float = height
        self.neighbor_wall_ids_at_start: list = neighbor_wall_ids_at_start
        self.neighbor_wall_ids_at_end: list = neighbor_wall_ids_at_end

    def generate_mesh(self):
        # Calculate the direction from start to end
        start_point = np.array(self.wall.get_min_bound())
        end_point = np.array(self.wall.get_max_bound())
        direction = end_point - start_point
        direction /= np.linalg.norm(direction)

        # Calculate the orthogonal directions to the main direction
        orthogonal1 = np.cross(direction, np.array([1, 0, 0]))
        if np.linalg.norm(orthogonal1) < 1e-5:
            orthogonal1 = np.cross(direction, np.array([0, 1, 0]))
        orthogonal1 /= np.linalg.norm(orthogonal1)
        orthogonal2 = np.cross(direction, orthogonal1)
        orthogonal2 /= np.linalg.norm(orthogonal2)

        # Calculate the vertices of the mesh
        vertices = np.array([
            start_point - self.height / 2 * orthogonal1 - self.width / 2 * orthogonal2,
            start_point - self.height / 2 * orthogonal1 + self.width / 2 * orthogonal2,
            start_point + self.height / 2 * orthogonal1 - self.width / 2 * orthogonal2,
            start_point + self.height / 2 * orthogonal1 + self.width / 2 * orthogonal2,
            end_point - self.height / 2 * orthogonal1 - self.width / 2 * orthogonal2,
            end_point - self.height / 2 * orthogonal1 + self.width / 2 * orthogonal2,
            end_point + self.height / 2 * orthogonal1 - self.width / 2 * orthogonal2,
            end_point + self.height / 2 * orthogonal1 + self.width / 2 * orthogonal2,
        ])

        # Define the triangles of the mesh
        triangles = np.array([
            [0, 1, 2],
            [1, 3, 2],
            [4, 6, 5],
            [5, 6, 7],
            [0, 4, 1],
            [1, 4, 5],
            [2, 3, 6],
            [3, 7, 6],
            [0, 2, 4],
            [2, 6, 4],
            [1, 5, 3],
            [3, 5, 7],
        ])

        # Create the mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.paint_uniform_color([0.5, 0.124, 0.4])
        return mesh


# {
#     "id": 367243,
#     "start_pt": [
#         23.790154066836081,
#         -27.03618027157281,
#         0.0
#     ],
#     "end_pt": [
#         -20.737415322435009,
#         -27.03618027157281,
#         0.0
#     ],
#     "width": 0.2032,
#     "height": 2.8955999999995221,
#     "neighbor_wall_ids_at_start": [
#         369767
#     ],
#     "neighbor_wall_ids_at_end": [
#         394787
#     ]
# },
def read_walls_from_json(json_path: str) -> list[Wall]:
    with open(json_path, "r") as file:
        data = json.load(file)
        walls = []
        for wall_data in data:
            wall = Wall(
                id=wall_data["id"],
                wall=o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=wall_data["start_pt"],
                    max_bound=wall_data["end_pt"]
                ),
                width=wall_data["width"],
                height=wall_data["height"],
                neighbor_wall_ids_at_start=wall_data["neighbor_wall_ids_at_start"],
                neighbor_wall_ids_at_end=wall_data["neighbor_wall_ids_at_end"]
            )
            walls.append(wall)
        return walls


if __name__ == "__main__":
    walls = read_walls_from_json("/home/lzq/Downloads/train/json_train/05_MedOffice_01_F2_walls.json")
    for wall in walls:
        bbox = wall.wall.get_axis_aligned_bounding_box()
        wall.wall = o3d.geometry.AxisAlignedBoundingBox(
            np.array([bbox.get_min_bound()[0], bbox.get_min_bound()[1], 3]),
            np.array([bbox.get_max_bound()[0], bbox.get_max_bound()[1], 3])
        )
        wall.height = 0.1
    wall_mesh_list = []
    for wall in walls:
        wall_mesh_list.append(wall.generate_mesh())
    plypath = "/home/lzq/Desktop/LAZ_test/08_ShortOffice_01_F2.ply"
    pcd = o3d.io.read_point_cloud(plypath)
    wall_mesh_list.append(pcd)
    o3d.visualization.draw_geometries(wall_mesh_list)
