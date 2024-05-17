#using open3d to convert laz to ply
import open3d as o3d
import laspy
import numpy as np
import os

def laz2ply(laz_path: str, ply_path: str):
    las_file = laspy.read(laz_path)
    points = np.vstack((las_file.x, las_file.y, las_file.z)).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    #voxal downsample and calculate density
    o3d.utilit


    o3d.io.write_point_cloud(ply_path, pcd)


if __name__ == "__main__":
    Path = "/home/lzq/Desktop/LAZ_test"

    for file in os.listdir(Path):
        if file.endswith(".laz"):
            print(f"processing {file}")
            laz2ply(f"{Path}/{file}", f"{Path}/{os.path.splitext(file)[0]}.ply")