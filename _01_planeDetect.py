import open3d as o3d
import numpy as np
import os

from open3d.cuda.pybind.geometry import AxisAlignedBoundingBox
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import heightlowmap
import json
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from scipy.ndimage import convolve
from scipy.stats import trim_mean, trimboth
from typing import Dict, Union, Tuple

OUTPUT_PREFIX = "output/output_data/"


# don`t touch magic number here
# we test them and find the value fit the best
# this program was writen by lzq, who hold a bachelor degree in art&design
# I try my best to add comments and type hints
# in order to help you understand the code
# feel free to contact me if you have any question: ziq93812@gmail.com

class CustomError(Exception):
    pass


def plot_stability_scores(scores):
    plt.hist(scores, bins='auto')
    plt.title('Stability Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.show()


def getName(path: str) -> str:
    name = os.path.basename(path)
    names = name.split('_')[0:4]
    return '_'.join(names)


class Subimage:
    def __init__(self, bias_x: int, bias_y: int, width: int, height: int, img: np.ndarray):
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.width = width
        self.height = height
        self.img = img
        self.filled = False

    def fill_img(self, img: np.ndarray):
        self.filled = True
        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                if i + self.bias_y >= img.shape[0] or j + self.bias_x >= img.shape[1]:
                    continue
                if self.img[i, j] == True:
                    img[i + self.bias_y, j + self.bias_x] = 255


def compare_oboxes(obox):
    min_b = obox.get_min_bound()
    max_b = obox.get_max_bound()
    return (max_b[0] - min_b[0]) * (max_b[1] - min_b[1])


def rel2abs_Path(relativePath: str) -> str:
    return f"{os.path.dirname(os.path.abspath(__file__))}/{relativePath}"


def getBaseNameWithoutExtension(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def to_fill(subimage: Subimage, res: float) -> bool:
    if subimage.filled:
        return False
    if subimage.width * res < 0.3 or subimage.height * res < 0.46:
        return True
    if subimage.width * res < 0.6 and subimage.height < 0.6:
        return True


def extract_1subimage(img: np.ndarray, x: int, y: int) -> Subimage:
    height, width = img.shape
    visited = np.zeros((height, width), dtype=bool)
    min_x, min_y, max_x, max_y = width, height, 0, 0

    stack = []
    stack.append((y, x))
    while len(stack) > 0:
        y, x = stack.pop()
        if not (0 <= x < width and 0 <= y < height) or visited[y, x] or img[y, x] != 0:
            continue
        visited[y, x] = True
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        img[y, x] = 255
        stack.extend([
            (y + 1, x),
            (y - 1, x),
            (y, x + 1),
            (y, x - 1),
            (y + 1, x + 1),
            (y + 1, x - 1),
            (y - 1, x + 1),
            (y - 1, x - 1)])

    # get the subimage
    subimage = Subimage(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1, visited[min_y:max_y + 1, min_x:max_x + 1])
    return subimage


# Warning: bug contained, if the image is too large, the function will not work
def binarize(img: np.ndarray) -> np.ndarray:
    pixels = img.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(pixels)
    out = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] < 0.001:
                out[i, j] = 0
            else:
                out[i, j] = 255
    return out


def find_max(indiceList: list[int], point_cloud: o3d.geometry.PointCloud):
    max_value = point_cloud.points[indiceList[0]][2]
    for num in indiceList:
        if point_cloud.points[num][2] > max_value:
            max_value = point_cloud.points[num][2]
    return max_value


def find_min(indiceList: list[int], point_cloud: o3d.geometry.PointCloud):
    min_value = point_cloud.points[indiceList[0]][2]
    for num in indiceList:
        if point_cloud.points[num][2] < min_value:
            min_value = point_cloud.points[num][2]
    return min_value


# not used
def newDetect(
        point_cloud: o3d.geometry.PointCloud, visualize: bool, name: str):
    res = 0.1
    height = 0.1
    # Convert the points to a NumPy array
    points = np.array(point_cloud.points)

    # Compute the x and y indices
    x_indices = (points[:, 0] / res).astype(int)
    y_indices = (points[:, 1] / res).astype(int)

    # Create a dictionary to store the indices
    pdict = defaultdict(list)

    # Populate the dictionary
    for i, (x_index, y_index) in enumerate(zip(x_indices, y_indices)):
        pdict[(x_index, y_index)].append(i)
    heights = []
    indices = []
    for key in pdict.keys():
        # get max and min in each list

        points = np.array(point_cloud.points)[:, 2][pdict[key]]
        max_z = np.max(points)
        min_z = np.min(points)
        mask = (points > min_z + height) & (points < max_z - height)
        res = np.array(pdict[key])[mask]
        indices.extend(res)
        heights.extend(points)

    print(len(indices))

    filtered_point_cloud: o3d.geometry.PointCloud = point_cloud.select_by_index(indices)
    # show point cloud
    if visualize:
        o3d.visualization.draw_geometries([filtered_point_cloud])

    plot_stability_scores(heights)
    plt.savefig(rel2abs_Path("{}_heights.png".format(OUTPUT_PREFIX + getName(name))))
    raise CustomError("finished,continue next task")


def get_floor_ceiling(name: str, point_cloud: o3d.geometry.PointCloud, visualize: bool, useNewDetect: bool) \
        -> tuple[AxisAlignedBoundingBox, AxisAlignedBoundingBox]:
    pass
    # step 1: filter the point cloud using normals ---------------------------------------------------------------------
    normal_criteria = np.array([0, 0, 1])
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.array(point_cloud.normals)
    dot_products = np.dot(normals, normal_criteria)
    indices = np.where((dot_products < 0.9) & (dot_products > -0.9))[0]
    filtered_point_cloud: o3d.geometry.PointCloud = point_cloud.select_by_index(indices)

    pass
    # step 2: generate a 3d array filter noise, get x, y bounding ------------------------------------------------------
    res = 0.4
    filtered_bbox_max = filtered_point_cloud.get_max_bound()
    filtered_bbox_min = filtered_point_cloud.get_min_bound()
    map_w = int((filtered_bbox_max[0] - filtered_bbox_min[0]) / res + 1)
    map_h = int((filtered_bbox_max[1] - filtered_bbox_min[1]) / res + 1)
    map_d = int((filtered_bbox_max[2] - filtered_bbox_min[2]) / res + 1)
    count_map = np.zeros((map_w, map_h, map_d))
    size = len(filtered_point_cloud.points)
    i = 0
    while i < size:
        p = filtered_point_cloud.points[i]
        i += 320
        x_index = int((p[0] - filtered_bbox_min[0]) / (filtered_bbox_max[0] - filtered_bbox_min[0]) * (map_w - 1))
        y_index = int((p[1] - filtered_bbox_min[1]) / (filtered_bbox_max[1] - filtered_bbox_min[1]) * (map_h - 1))
        z_index = int((p[2] - filtered_bbox_min[2]) / (filtered_bbox_max[2] - filtered_bbox_min[2]) * (map_d - 1))
        count_map[x_index, y_index, z_index] = 1
    kernel = np.ones((15, 15, 3))
    conv_result = convolve(count_map, kernel, mode='constant', cval=0.0)
    conv_result_max = np.max(conv_result)
    value = np.percentile(conv_result[(conv_result >= 25) & (conv_result <= conv_result_max - 25)], 70)
    indices = np.argwhere(conv_result >= value)  # pow(res*10,2)*(40/16))#magic number, fit algrathm well
    min_indices = indices.min(axis=0)
    max_indices = indices.max(axis=0) + 1
    xmin = min_indices[0] * res + filtered_bbox_min[0]
    ymin = min_indices[1] * res + filtered_bbox_min[1]
    xmax = max_indices[0] * res + filtered_bbox_min[0]
    ymax = max_indices[1] * res + filtered_bbox_min[1]

    pass
    # step 3: extract walls, use the altitude of walls as the altitude of ceiling and floor ----------------------------
    if not useNewDetect:

        # step 3.1: calculate density map and histogram for debug ------------------------------------------------------
        conv_result = np.sum(conv_result, axis=2)
        heights = np.asarray(filtered_point_cloud.points)[::, 2]  # replace with your data

        plt.hist(heights, edgecolor='black')
        plt.savefig(rel2abs_Path("{}_histogram.png".format(OUTPUT_PREFIX + getName(name))))
        plt.close()
        fig, ax = plt.subplots()
        ax.imshow(conv_result, cmap='hot')
        rect = patches.Rectangle(
            (min_indices[1], min_indices[0]),
            max_indices[1] - min_indices[1],
            max_indices[0] - min_indices[0],
            edgecolor='w',
            facecolor='none')
        ax.add_patch(rect)
        plt.savefig(rel2abs_Path("{}_densitymap.png".format(OUTPUT_PREFIX + getName(name))))
        plt.close()
        # step 3.2: detect the walls -----------------------------------------------------------------------------------
        filtered_point_cloud = filtered_point_cloud.crop(
            o3d.geometry.AxisAlignedBoundingBox(
                min_bound=(xmin, ymin, -np.inf),
                max_bound=(xmax, ymax, np.inf)
            )
        )
        oboxes = point_cloud.detect_planar_patches(
            normal_variance_threshold_deg=60,
            coplanarity_deg=60,
            outlier_ratio=3,
            min_plane_edge_length=0.5,
            min_num_points=75,
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=100))
        print("Detected {} patches".format(len(oboxes)))

        if len(oboxes) < 2:
            oboxes = point_cloud.detect_planar_patches(
                normal_variance_threshold_deg=60,
                coplanarity_deg=60,
                outlier_ratio=1,
                min_plane_edge_length=2,
                min_num_points=25,
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
            print("Detected {} patches".format(len(oboxes)))
            if len(oboxes) < 2:
                raise CustomError("Not enough patches detected, consider change parameters")
        sorted_oboxes = sorted(oboxes, key=compare_oboxes)
        # Assuming sorted_oboxes is your sorted list
        vertical_boxes = []

        # step 3.3: assume the height of walls are the altitude to ceilings and floors ---------------------------------
        for box in sorted_oboxes:
            # Assuming the coordinates are stored in box.coordinates
            min_bound = box.get_min_bound()
            max_bound = box.get_max_bound()

            x_diff = max_bound[0] - min_bound[0]
            y_diff = max_bound[1] - min_bound[1]
            z_diff = max_bound[2] - min_bound[2]
            # wall higher than 5m or lower than 2m are not reasonable
            if z_diff > 5:
                continue
            if z_diff < 2:
                continue
            if z_diff > x_diff or z_diff > y_diff:
                # Assuming `rotation_matrix` is your rotation matrix
                local_up_vector = np.array([0, 0, 1])

                global_up_vector = np.dot(box.R, local_up_vector)

                dot_products = np.dot(global_up_vector, local_up_vector)
                if 0.1 > dot_products > -0.1:
                    vertical_boxes.append(box)
        print(f"Found {len(vertical_boxes)} vertical boxes")
        z_values_floor = np.array([box.get_min_bound()[2] for box in vertical_boxes])
        winsorized_mean_floor = np.mean(trimboth(z_values_floor, 0.4))  # Winsorize 10% at both ends
        z_values_ceiling = np.array([box.get_max_bound()[2] for box in vertical_boxes])
        winsorized_mean_ceiling = np.mean(trimboth(z_values_ceiling, 0.4))  # Winsorize 10% at both ends

        # step 3.4: visualize ------------------------------------------------------------------------------------------
        if visualize:
            geometries = []
            for obox in vertical_boxes:
                if isinstance(obox, o3d.geometry.OrientedBoundingBox):
                    obox.get_min_bound()
                    obox.get_max_bound()
                    mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.001])
                    mesh.paint_uniform_color(obox.color)
                    geometries.append(mesh)
                    geometries.append(obox)
            geometries.append(filtered_point_cloud)
            o3d.visualization.draw_geometries(geometries)
        # extend the floor and ceiling

    pass
    # step 4: combine x, y values and z altitudes ----------------------------------------------------------------------
    extend_floor = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(xmin, ymin, winsorized_mean_floor - 1),
        max_bound=(xmax, ymax, winsorized_mean_floor + 0.3))

    extend_ceiling = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(xmin, ymin, winsorized_mean_ceiling - 1),
        max_bound=(xmax, ymax, winsorized_mean_ceiling + 3))

    print(extend_floor.get_min_bound(), extend_floor.get_max_bound())
    print(extend_ceiling.get_min_bound(), extend_ceiling.get_max_bound())
    print()

    return extend_floor, extend_ceiling


def extract_planes_point(
        pcd: o3d.geometry.PointCloud,
        floor: o3d.geometry.OrientedBoundingBox,
        ceiling: o3d.geometry.OrientedBoundingBox,
        visualize: bool = False) \
        -> tuple[o3d.geometry.PointCloud
        , o3d.geometry.PointCloud]:
    floor_point_cloud = pcd.crop(floor)
    ceiling_point_cloud = pcd.crop(ceiling)

    return floor_point_cloud, ceiling_point_cloud


# main function here
def process_pc(cloud_path: str, visualize: bool = False, res: float = 0.15, usenewDetect: bool = False) -> None:
    print("process_pc:--------------------------------------------------")
    print(cloud_path)

    pass
    # step 1: read the point cloud -------------------------------------------------------------------------------------
    point_cloud: o3d.geometry.PointCloud = o3d.io.read_point_cloud(cloud_path)
    point_number: int = len(point_cloud.points)
    if not point_cloud.has_points():
        raise CustomError("No points in the point cloud")
    # GET FILE NAME without extension
    file_name: str = getName(cloud_path)
    # normal is not correct, we need to recalculate it
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    pass
    # step 2: extract floor and ceiling --------------------------------------------------------------------------------
    floor_bb, ceiling_bb = get_floor_ceiling(cloud_path, point_cloud, visualize, usenewDetect)
    floor_pcd, ceiling_pcd = extract_planes_point(point_cloud, floor_bb, ceiling_bb, visualize)

    # step 3: generate floor and ceiling height map --------------------------------------------------------------------
    floor_high, floor_low = heightlowmap.get_high_low_img(floor_pcd, floor_bb, res)
    ceiling_high, ceiling_low = heightlowmap.get_high_low_img(ceiling_pcd, ceiling_bb, res)

    pass
    # step 4: generate mask for denoise used later ---------------------------------------------------------------------
    kernel3: np.ndarray = np.ones((3, 3), np.uint8)
    kernel5: np.ndarray = np.ones((5, 5), np.uint8)
    kernel7: np.ndarray = np.ones((7, 7), np.uint8)
    ceiling_high_process: np.ndarray = ceiling_high / np.max(ceiling_high) * 255
    ceiling_high_process: np.ndarray = cv2.erode(ceiling_high_process, kernel3, iterations=1)
    if len(ceiling_high_process.shape) == 3 and ceiling_high_process.shape[2] == 3:
        # Image is color
        img_gray: np.ndarray = cv2.cvtColor(ceiling_high_process, cv2.COLOR_BGR2GRAY)
    else:
        # Image is already grayscale
        img_gray: np.ndarray = ceiling_high_process
    img_gray: np.ndarray = cv2.dilate(img_gray, kernel7, iterations=1)
    img_8bit: np.ndarray = cv2.convertScaleAbs(img_gray)
    num_labels, labels = cv2.connectedComponents(img_8bit)
    blob_sizes: np.ndarray = np.bincount(labels.flatten())
    # The first blob is the background, so ignore it
    blob_sizes[0] = 0
    largest_blob: int = blob_sizes.argmax()
    largest_blob_mask: np.ndarray = np.uint8(labels == largest_blob) * 255

    pass
    # step 5: save the data --------------------------------------------------------------------------------------------
    point_cloud: o3d.geometry.PointCloud = o3d.io.read_point_cloud(cloud_path)
    point_number: int = len(point_cloud.points)
    crop_bbox: o3d.geometry.AxisAlignedBoundingBox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(floor_bb.get_min_bound()[0], floor_bb.get_min_bound()[1], floor_bb.get_min_bound()[2] + 1),
        max_bound=(ceiling_bb.get_max_bound()[0], ceiling_bb.get_max_bound()[1], ceiling_bb.get_min_bound()[2] + 1)
    )
    point_cloud_denoise: o3d.geometry.PointCloud = point_cloud.crop(crop_bbox)

    bb_data: Dict[str, Union[Dict[str, list[float]], float, int]] = {
        "floor": {"min": [floor_bb.get_min_bound()[0], floor_bb.get_min_bound()[1], floor_bb.get_min_bound()[2]]
            , "max": [floor_bb.get_max_bound()[0], floor_bb.get_max_bound()[1], floor_bb.get_max_bound()[2]]},
        "ceiling": {"min": [ceiling_bb.get_min_bound()[0], ceiling_bb.get_min_bound()[1], ceiling_bb.get_min_bound()[2]]
            , "max": [ceiling_bb.get_max_bound()[0], ceiling_bb.get_max_bound()[1], ceiling_bb.get_max_bound()[2]]},
        "x_size": floor_bb.get_max_bound()[0] - floor_bb.get_min_bound()[0],
        "y_size": floor_bb.get_max_bound()[1] - floor_bb.get_min_bound()[1],
        "z_size": ceiling_bb.get_min_bound()[2] - floor_bb.get_min_bound()[2],
        "point_number": point_number,
        "noise_rate": (point_number - len(point_cloud_denoise.points)) / point_number,
    }
    with open("{}_bbox.json".format(OUTPUT_PREFIX + file_name), 'w') as f:
        json.dump(bb_data, f)
    print("saving images")
    cv2.imwrite(rel2abs_Path("{}_ceiling_mask.png".format(OUTPUT_PREFIX + file_name)), largest_blob_mask)
    cv2.imwrite(rel2abs_Path("{}_floor_high_img.png".format(OUTPUT_PREFIX + file_name)), floor_high)
    cv2.imwrite(rel2abs_Path("{}_floor_low_img.png".format(OUTPUT_PREFIX + file_name)), floor_low)
    cv2.imwrite(rel2abs_Path("{}_ceiling_high_img.png".format(OUTPUT_PREFIX + file_name)), ceiling_high)
    cv2.imwrite(rel2abs_Path("{}_ceiling_low_img.png".format(OUTPUT_PREFIX + file_name)), ceiling_low)
    print("done")
    print("")


Path = "/media/lzq/Windows/Users/14318/scan2bim2024/3d/test/5cm"
if __name__ == "__main__":
    # process_pc("/media/lzq/Windows/Users/14318/scan2bim2024/2d/test/2cm/25_Parking_01_F2_s0p01m.ply", False, 0.02,False)
    # process_pc("/media/lzq/Windows/Users/14318/scan2bim2024/2d/test/2cm/02_TallOffice_01_F7_s0p01m.ply", False, 0.02,False)
    # exit(0)
    # get all clouds in the folder
    from multiprocessing import Pool
    import os
    import argparse


    def process_file(file):
        if file.endswith(".ply"):
            try:
                process_pc(f"{Path}/{file}", False, 0.05, False)
            except CustomError as e:
                print(e)


    parser = argparse.ArgumentParser()
    parser.add_argument("Path", help="Path to the directory containing .ply files")
    parser.add_argument("file", help="File to process")
    args = parser.parse_args()

    Path = args.Path
    file = args.file

    process_file(file)  # s0p01m
    # if __name__ == "__main__":
    #     with Pool(os.cpu_count()) as p:
    #         p.map(process_file, os.listdir(Path))
# 11_MedOffice_05_F4 have problem --> solved
# 25_Parking_01_F1 floor detection problem -> ceiling solved floor unsolved -->solved
# 25_Parking_01_F2 floor detection problem  -->solved
