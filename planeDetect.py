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
def getName(path: str)->str:
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
def process_pc(cloud_path: str, visualize: bool = False, res: float = 0.15):
    point_cloud: o3d.geometry.PointCloud = o3d.io.read_point_cloud(cloud_path)
    # GET FILE NAME without extension
    file_name = getName(cloud_path)
    if (not point_cloud.has_normals()):
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    floor_bb, ceiling_bb = get_floor_ceiling(cloud_path, point_cloud, visualize)
    # print these to json file
    bb_data = {
        "floor": {"min": [floor_bb.get_min_bound()[0], floor_bb.get_min_bound()[1], floor_bb.get_min_bound()[2]]
            , "max": [floor_bb.get_max_bound()[0], floor_bb.get_max_bound()[1], floor_bb.get_max_bound()[2]]},
        "ceiling": {"min": [ceiling_bb.get_min_bound()[0], ceiling_bb.get_min_bound()[1], ceiling_bb.get_min_bound()[2]]
            , "max": [ceiling_bb.get_max_bound()[0], ceiling_bb.get_max_bound()[1], ceiling_bb.get_max_bound()[2]]}
    }
    with open("output_data_2d/{}_bbox.json".format(file_name), 'w') as f:
        json.dump(bb_data, f)
    floor_pcd, ceiling_bb = extract_planes_point(point_cloud, floor_bb, ceiling_bb, visualize)

    floor_high, floor_low = heightlowmap.get_high_low_img(floor_pcd, res)
    ceiling_high, ceiling_low = heightlowmap.get_high_low_img(ceiling_bb, res)
    # save the images
    print("saving images")
    cv2.imwrite(rel2abs_Path("output_data_2d/{}_floor_high_img.png".format(file_name)), floor_high)
    cv2.imwrite(rel2abs_Path("output_data_2d/{}_floor_low_img.png".format(file_name)), floor_low)
    cv2.imwrite(rel2abs_Path("output_data_2d/{}_ceiling_high_img.png".format(file_name)), ceiling_high)
    cv2.imwrite(rel2abs_Path("output_data_2d/{}_ceiling_low_img.png".format(file_name)), ceiling_low)

    # cv use 0~1 while plt con`t care
    # cv2.imwrite(rel2abs_Path("output_data/{}_ceiling_low_bi_img.png".format(file_name)), ceiling_low_bi)
    #
    # tempcopy = np.copy(ceiling_low_bi)
    # subimages = []
    # # get all zero subimages and save them
    # for y in range(tempcopy.shape[0]):
    #     for x in range(tempcopy.shape[1]):
    #         if tempcopy[y, x] == 0:
    #             subimages.append(extract_1subimage(tempcopy, x, y))
    #
    # print(f"extracting {len(subimages)}th subimage")
    # for i, subimage in enumerate(subimages):
    #     if to_fill(subimage, res):
    #         print(f"filling {i}th subimage")
    #         subimage.fill_img(ceiling_low_bi)
    #         subimage.filled = True
    # cv2.imwrite(rel2abs_Path("output_data/{}_ceiling_low_bi_fill_img.png".format(file_name)), ceiling_low_bi)


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


def get_floor_ceiling(name: str, point_cloud: o3d.geometry.PointCloud, visualize: bool) \
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
    
    count_map = np.zeros((map_w, map_h))
    for i, p in enumerate(filtered_point_cloud.points):
        if (i % 64 != 0):
            continue
        x_index = int((p[0] - filtered_bbox_min[0]) / (filtered_bbox_max[0] - filtered_bbox_min[0]) * (map_w - 1))
        y_index = int((p[1] - filtered_bbox_min[1]) / (filtered_bbox_max[1] - filtered_bbox_min[1]) * (map_h - 1))
        count_map[x_index, y_index] += 1

    indices = np.argwhere(count_map > 50)
    min_indices = indices.min(axis=0)
    max_indices = indices.max(axis=0) + 1
    xmin = min_indices[0] * res + filtered_bbox_min[0]
    ymin = min_indices[1] * res + filtered_bbox_min[1]
    xmax = max_indices[0] * res + filtered_bbox_min[0]
    ymax = max_indices[1] * res + filtered_bbox_min[1]
    
    plt.hist(count_map, edgecolor='black')
    plt.savefig(rel2abs_Path("output_data_2d/{}_histogram.png".format(getName(name))))
    plt.close()

    fig, ax = plt.subplots()
    ax.imshow(count_map, cmap='hot')
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
            
            
    oboxes = filtered_point_cloud.detect_planar_patches(
        normal_variance_threshold_deg=35,
        coplanarity_deg=60,
        outlier_ratio=3,
        min_plane_edge_length=4,
        min_num_points=400,
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

    # extend the floor and ceiling
    floor: o3d.geometry.AxisAlignedBoundingBox = floor.get_axis_aligned_bounding_box()
    ceiling: o3d.geometry.AxisAlignedBoundingBox = ceiling.get_axis_aligned_bounding_box()

    
    extend_floor = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(xmin, ymin, floor.get_min_bound()[2] - 0.05),
        max_bound=(xmax, ymax, floor.get_max_bound()[2] + 0.05))

    extend_ceiling = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(xmin, ymin, ceiling.get_min_bound()[2] - 0.3),
        max_bound=(xmax, ymax, ceiling.get_max_bound()[2] + 3))
    
    print("Floor:--------------------------------------------------")
    print(name)
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


Path = "/media/lzq/Windows/Users/14318/scan2bim2024/2d/test/2cm"
if __name__ == "__main__":
    process_pc("/media/lzq/Windows/Users/14318/scan2bim2024/2d/test/2cm/02_TallOffice_01_F13_s0p01m.ply", True, res=0.02)
    # get all clouds in the folder
    for file in os.listdir(Path):
        if file.endswith(".ply"):
            process_pc(f"{Path}/{file}", False, res=0.02)
# 11_MedOffice_05_F4 have problem --> solved
# 25_Parking_01_F1 floor detection problem -> ceiling solved floor unsolved -->solved
# 25_Parking_01_F2 floor detection problem  -->solved
