import open3d as o3d
import numpy as np
import os
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


# don`t touch magic number here
# it test them and find the value fit the best


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


# main function here
def process_pc(cloud_path: str, visualize: bool = False, res: float = 0.15, usenewDetect: bool = False):
    print("process_pc:--------------------------------------------------")
    print(cloud_path)
    point_cloud: o3d.geometry.PointCloud = o3d.io.read_point_cloud(cloud_path)
    if not point_cloud.has_points():
        raise CustomError("No points in the point cloud")
    # GET FILE NAME without extension
    file_name = getName(cloud_path)
    if (not point_cloud.has_normals()):
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    floor_bb, ceiling_bb = get_floor_ceiling(cloud_path, point_cloud, visualize, usenewDetect)
    # print these to json file
    bb_data = {
        "floor": {"min": [floor_bb.get_min_bound()[0], floor_bb.get_min_bound()[1], floor_bb.get_min_bound()[2]]
            , "max": [floor_bb.get_max_bound()[0], floor_bb.get_max_bound()[1], floor_bb.get_max_bound()[2]]},
        "ceiling": {"min": [ceiling_bb.get_min_bound()[0], ceiling_bb.get_min_bound()[1], ceiling_bb.get_min_bound()[2]]
            , "max": [ceiling_bb.get_max_bound()[0], ceiling_bb.get_max_bound()[1], ceiling_bb.get_max_bound()[2]]}
    }
    with open("output_data_2d/{}_bbox.json".format(file_name), 'w') as f:
        json.dump(bb_data, f)
    floor_pcd, ceiling_pcd = extract_planes_point(point_cloud, floor_bb, ceiling_bb, visualize)

    floor_high, floor_low = heightlowmap.get_high_low_img(floor_pcd, floor_bb, res)
    ceiling_high, ceiling_low = heightlowmap.get_high_low_img(ceiling_pcd, ceiling_bb, res)
    #find the largest group of points
    # Assuming `img` is your image

    kernel = np.ones((9, 9), np.uint8)
    floor_high = cv2.dilate(floor_high, kernel, iterations=2)
    #floor_low = cv2.erode(floor_low, kernel, iterations=1)
    plt.imshow(floor_high)
    plt.show()
    plt.close()
    if len(floor_high.shape) == 3 and floor_high.shape[2] == 3:
        # Image is color
        img_gray = cv2.cvtColor(floor_high, cv2.COLOR_BGR2GRAY)
    else:
        # Image is already grayscale
        img_gray = floor_high
    img_8bit = cv2.convertScaleAbs(img_gray)
    num_labels, labels = cv2.connectedComponents(img_8bit)
    blob_sizes = np.bincount(labels.flatten())
    # The first blob is the background, so ignore it
    blob_sizes[0] = 0
    largest_blob = blob_sizes.argmax()
    largest_blob_mask = np.uint8(labels == largest_blob)
    plt.imshow(largest_blob_mask)
    plt.show()
    plt.close()
    # save the images
    print("saving images")
    cv2.imwrite(rel2abs_Path("output_data_2d/{}_floor_high_img.png".format(file_name)), floor_high)
    cv2.imwrite(rel2abs_Path("output_data_2d/{}_floor_low_img.png".format(file_name)), floor_low)
    cv2.imwrite(rel2abs_Path("output_data_2d/{}_ceiling_high_img.png".format(file_name)), ceiling_high)
    cv2.imwrite(rel2abs_Path("output_data_2d/{}_ceiling_low_img.png".format(file_name)), ceiling_low)
    print("done")
    print("")


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


def newDetect(
        point_cloud: o3d.geometry.PointCloud, visualize: bool, name: str
) -> tuple[
    o3d.geometry.AxisAlignedBoundingBox,
    o3d.geometry.AxisAlignedBoundingBox
]:
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
    plt.savefig(rel2abs_Path("output_data_2d/{}_heights.png".format(getName(name))))
    raise CustomError("finished,continue next task")


def get_floor_ceiling(name: str, point_cloud: o3d.geometry.PointCloud, visualize: bool, useNewDetect: bool) \
        -> tuple[o3d.geometry.OrientedBoundingBox, o3d.geometry.OrientedBoundingBox]:
    normal_criteria = np.array([0, 0, 1])  # This is an example, adjust it to your needs
    res = 0.4

    if not point_cloud.has_normals():
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    normals = np.array(point_cloud.normals)
    dot_products = np.dot(normals, normal_criteria)
    indices = np.where((dot_products > 0.9) | (dot_products < -0.9))[0]
    filtered_point_cloud: o3d.geometry.PointCloud = point_cloud.select_by_index(indices)

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
    value = np.percentile(conv_result[(conv_result >= 25) & (conv_result <= conv_result_max - 25)], 50)
    indices = np.argwhere(conv_result >= value)  # pow(res*10,2)*(40/16))#magic number, fit algrathm well
    min_indices = indices.min(axis=0)
    max_indices = indices.max(axis=0) + 1
    xmin = min_indices[0] * res + filtered_bbox_min[0]
    ymin = min_indices[1] * res + filtered_bbox_min[1]
    xmax = max_indices[0] * res + filtered_bbox_min[0]
    ymax = max_indices[1] * res + filtered_bbox_min[1]

    if not useNewDetect:

        conv_result = np.sum(conv_result, axis=2)
        heights = np.asarray(filtered_point_cloud.points)[::, 2]  # replace with your data

        plt.hist(heights, edgecolor='black')
        plt.savefig(rel2abs_Path("output_data_2d/{}_histogram.png".format(getName(name))))
        plt.close()
        fig, ax = plt.subplots()
        ax.imshow(conv_result, cmap='hot')
        rect = patches.Rectangle(
            (min_indices[1], min_indices[0]),
            max_indices[1] - min_indices[1],
            max_indices[0] - min_indices[0],
            edgecolor='r',
            facecolor='none')
        ax.add_patch(rect)
        plt.savefig(rel2abs_Path("output_data_2d/{}_densitymap.png".format(getName(name))))
        plt.close()
        filtered_point_cloud = filtered_point_cloud.crop(
            o3d.geometry.AxisAlignedBoundingBox(
                min_bound=(xmin, ymin, -np.inf),
                max_bound=(xmax, ymax, np.inf)
            )
        )

        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

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
                min_num_points=200,
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
            print("Detected {} patches".format(len(oboxes)))
            if len(oboxes) < 2:
                raise CustomError("Not enough patches detected, consider change parameters")
                return None, None
        sorted_oboxes = sorted(oboxes, key=compare_oboxes)
        # Assuming sorted_oboxes is your sorted list
        vertical_boxes = []

        for box in sorted_oboxes:
            # Assuming the coordinates are stored in box.coordinates
            min_bound = box.get_min_bound()
            max_bound = box.get_max_bound()

            x_diff = max_bound[0] - min_bound[0]
            y_diff = max_bound[1] - min_bound[1]
            z_diff = max_bound[2] - min_bound[2]
            if z_diff > 5:
                continue
            if z_diff < 2:
                continue
            if z_diff > x_diff or z_diff > y_diff:
                # Assuming `rotation_matrix` is your rotation matrix
                local_up_vector = np.array([0, 0, 1])

                global_up_vector = np.dot(box.R, local_up_vector)

                dot_products = np.dot(global_up_vector, local_up_vector)
                if dot_products < 0.1 and dot_products > -0.1:
                    vertical_boxes.append(box)

        z_values_floor = np.array([box.get_min_bound()[2] for box in vertical_boxes])
        median_floor = np.median(z_values_floor)
        truncated_mean_floor = trim_mean(z_values_floor, 0.4)  # Truncate 10% at both ends
        winsorized_mean_floor = np.mean(trimboth(z_values_floor, 0.4))  # Winsorize 10% at both ends
        z_values_ceiling = np.array([box.get_max_bound()[2] for box in vertical_boxes])
        median_ceiling = np.median(z_values_ceiling)
        truncated_mean_ceiling = trim_mean(z_values_ceiling, 0.4)  # Truncate 10% at both ends
        winsorized_mean_ceiling = np.mean(trimboth(z_values_ceiling, 0.4))  # Winsorize 10% at both ends

        print(f"Found {len(vertical_boxes)} vertical boxes")
        max1: o3d.geometry.OrientedBoundingBox = sorted_oboxes[-1]  # Maximum value
        max2: o3d.geometry.OrientedBoundingBox = sorted_oboxes[-2]  # Second maximum value
        if max1.get_max_bound()[2] + max1.get_min_bound()[2] < max2.get_max_bound()[2] + max2.get_min_bound()[2]:
            floor = max1
            ceiling = max2
        else:
            floor = max2
            ceiling = max1
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
        floor: o3d.geometry.AxisAlignedBoundingBox = floor.get_axis_aligned_bounding_box()
        ceiling: o3d.geometry.AxisAlignedBoundingBox = ceiling.get_axis_aligned_bounding_box()
    pass

    if useNewDetect:
        floor, ceiling = newDetect(point_cloud, visualize, name)

    extend_floor = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(xmin, ymin, winsorized_mean_floor - 0.7),
        max_bound=(xmax, ymax, winsorized_mean_floor + 0.3))

    extend_ceiling = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(xmin, ymin, winsorized_mean_ceiling - 0.5),
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


Path = "/media/lzq/Windows/Users/14318/scan2bim2024/2d/test/5cm"
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


    # parser = argparse.ArgumentParser()
    # parser.add_argument("Path", help="Path to the directory containing .ply files")
    # parser.add_argument("file", help="File to process")
    # args = parser.parse_args()
    #
    # Path = args.Path
    # file = args.file

    process_file("14_Garage_01_F1_s0p01m.ply")#s0p01m
    # if __name__ == "__main__":
    #     with Pool(os.cpu_count()) as p:
    #         p.map(process_file, os.listdir(Path))
# 11_MedOffice_05_F4 have problem --> solved
# 25_Parking_01_F1 floor detection problem -> ceiling solved floor unsolved -->solved
# 25_Parking_01_F2 floor detection problem  -->solved
