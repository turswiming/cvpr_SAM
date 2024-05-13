import open3d as o3d
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import heightlowmap

Path = "input_data/02_TallOffice_01_F7_s0p01m.ply"


def compare_oboxes(obox):
    min_b = obox.get_min_bound()
    max_b = obox.get_max_bound()
    return (max_b[0] - min_b[0]) * (max_b[1] - min_b[1])


def rel2abs_Path(relativePath: str) -> str:
    return f"{os.path.dirname(os.path.abspath(__file__))}/{relativePath}"


# main function here
def process_pc(cloud_path: str, visualize: bool = False):
    point_cloud: o3d.geometry.PointCloud = o3d.io.read_point_cloud(cloud_path)
    assert (point_cloud.has_normals())

    floor_bb, ceiling_bb = get_floor_ceiling(point_cloud, False)

    floor_pcd, ceiling_bb = extract_planes_point(point_cloud, floor_bb, ceiling_bb, visualize)

    floor_high, floor_low = heightlowmap.get_high_low_img(floor_pcd, 0.05)
    ceiling_high, ceiling_low = heightlowmap.get_high_low_img(ceiling_bb, 0.05)
    # save the images
    plt.imsave(rel2abs_Path("output_data/floor_high_img.png"), floor_high)
    plt.imsave(rel2abs_Path("output_data/floor_low_img.png"), floor_low)
    plt.imsave(rel2abs_Path("output_data/ceiling_high_img.png"), ceiling_high)
    plt.imsave(rel2abs_Path("output_data/ceiling_low_img.png"), ceiling_low)

    # cv use 0~1 while plt con`t care
    cv2.imwrite(rel2abs_Path("output_data/ceiling_low_img_bi.png"), binarize(ceiling_low))


def binarize(img: np.ndarray) -> np.ndarray:
    pixels = img.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(pixels)

    # Replace each pixel with the centroid of its cluster
    segmented_image = kmeans.cluster_centers_[kmeans.labels_]
    segmented_image = segmented_image.reshape(img.shape)
    zeroclusterID = segmented_image[0, 0]
    for i in range(segmented_image.shape[0]):
        for j in range(segmented_image.shape[1]):
            if segmented_image[i, j] == zeroclusterID:
                segmented_image[i, j] = 0
            else:
                segmented_image[i, j] = 255
    return segmented_image


def get_floor_ceiling(point_cloud: o3d.geometry.PointCloud, visualize: bool) \
        -> tuple[o3d.geometry.OrientedBoundingBox, o3d.geometry.OrientedBoundingBox]:
    # using all defaults
    oboxes = point_cloud.detect_planar_patches(
        normal_variance_threshold_deg=60,
        coplanarity_deg=75,
        outlier_ratio=0.75,
        min_plane_edge_length=0,
        min_num_points=0,
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    print("Detected {} patches".format(len(oboxes)))
    sorted_oboxes = sorted(oboxes, key=compare_oboxes)
    # Assuming sorted_oboxes is your sorted list
    max1: o3d.geometry.OrientedBoundingBox = sorted_oboxes[-1]  # Maximum value
    max2: o3d.geometry.OrientedBoundingBox = sorted_oboxes[-2]  # Second maximum value
    if max1.get_max_bound()[2] + max1.get_min_bound()[2] < max2.get_max_bound()[2] + max2.get_min_bound()[2]:
        floor = max1
        ceiling = max2
    else:
        floor = max2
        ceiling = max1
    geometries = []
    for obox in (floor, ceiling):
        if isinstance(obox, o3d.geometry.OrientedBoundingBox):
            obox.get_min_bound()
            obox.get_max_bound()
            mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.001])
            mesh.paint_uniform_color(obox.color)
            geometries.append(mesh)
            geometries.append(obox)
    geometries.append(point_cloud)
    if visualize:
        o3d.visualization.draw_geometries(geometries)
    return floor, ceiling


def extract_planes_point(
        pcd: o3d.geometry.PointCloud,
        floor: o3d.geometry.OrientedBoundingBox,
        ceiling: o3d.geometry.OrientedBoundingBox,
        visualize: bool = False) \
        -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    floor_point_cloud = pcd.crop(floor)
    ceiling_point_cloud = pcd.crop(ceiling)

    return floor_point_cloud, ceiling_point_cloud


if __name__ == "__main__":
    process_pc(Path, True)
