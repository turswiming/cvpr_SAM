import os
import open3d as o3d
import numpy as np

#read all the ply files in the folder
no_room_path = "/media/lzq/Windows/Users/14318/scan2bim2024/3d/test/2cm_rooms/11_MedOffice_05_F2/non_rooms"
room_path = "/media/lzq/Windows/Users/14318/scan2bim2024/3d/test/2cm_rooms/11_MedOffice_05_F2"


#using open3d read these file
def read_ply_files_open3d(path):
    files = os.listdir(path)
    ply_files = []
    for file in files:
        if file.endswith(".ply"):
            ply_files.append(o3d.io.read_point_cloud(os.path.join(path, file)))
    return ply_files


no_room_ply = read_ply_files_open3d(no_room_path)
for file in no_room_ply:
    for color in file.colors:
        color *= np.array([0, 1, 1])
room_ply = read_ply_files_open3d(room_path)
for file in room_ply:
    for color in file.colors:
        color *= np.array([1, 0, 1])

#show these point clouds
o3d.visualization.draw_geometries(no_room_ply + room_ply)
