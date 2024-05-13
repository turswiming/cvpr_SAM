import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def get_high_low_img(
        point_cloud: o3d.geometry.PointCloud,
        resolution: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    axis_bb: o3d.geometry.AxisAlignedBoundingBox = point_cloud.get_axis_aligned_bounding_box()
    min_bb = axis_bb.get_min_bound()  # np.array[3,1]
    max_bb = axis_bb.get_max_bound()  # np.array[3,1]
    points = np.asarray(point_cloud.points)
    # Initialize an empty dictionary
    height_dict = {}
    lowest_dict = {}
    for point in points:
        # Convert the point to a tuple
        x = (point[0] - min_bb[0]) / resolution
        y = (point[1] - min_bb[1]) / resolution
        xycoord = (int(x), int(y))
        # Check if the point is already in the dictionary
        if xycoord in height_dict:
            height_dict[xycoord] = max(height_dict[xycoord], point[2])
        else:
            height_dict[xycoord] = point[2]

        if xycoord in lowest_dict:
            lowest_dict[xycoord] = min(lowest_dict[xycoord], point[2])
        else:
            lowest_dict[xycoord] = point[2]
    # convert these dict to image and save it
    height_img = np.zeros(
        (int((max_bb[0] - min_bb[0]) / resolution) + 1, int((max_bb[1] - min_bb[1]) / resolution) + 1))
    lowest_img = np.zeros(
        (int((max_bb[0] - min_bb[0]) / resolution) + 1, int((max_bb[1] - min_bb[1]) / resolution) + 1))
    for key, value in height_dict.items():
        height_img[key[0], key[1]] = (value - min_bb[2]) / (max_bb[2] - min_bb[2]) * 0.8 + 0.1
    for key, value in lowest_dict.items():
        lowest_img[key[0], key[1]] = (value - min_bb[2]) / (max_bb[2] - min_bb[2]) * 0.8 + 0.1
    return height_img, lowest_img


def main():
    cloudPath = "input_data/02_TallOffice_01_F7_s0p01m.ply"
    base_path = "/media/lzq/Windows/Users/14318/scan2bim2024"
    resolution = 0.02  #meters
    point_cloud: o3d.geometry.PointCloud = o3d.io.read_point_cloud(cloudPath)

    height, low = get_high_low_img(point_cloud, resolution)
    #save the images
    plt.imsave("height_img.png", height)
    plt.imsave("lowest_img.png", low)


if __name__ == "__main__":
    main()
