import numpy as np
import os
import json
import open3d as o3d
import matplotlib.pyplot as plt


def compute_intersection_matrix(group1, group2):
    # Initialize a 2D array to store the intersection counts
    confusion_matrix = np.zeros((len(group1), len(group2)), dtype=float)

    # For each mask in the first group...
    for i, mask1 in enumerate(group1):
        # For each mask in the second group...
        for j, mask2 in enumerate(group2):
            # Compute the intersection of the two masks and count the number of 'True' values
            intersection = np.logical_and(mask1, mask2)
            and_value = np.count_nonzero(intersection)
            lou = and_value / (np.count_nonzero(mask1) + np.count_nonzero(mask2) - and_value)
            confusion_matrix[i, j] = lou

    return confusion_matrix


def drawconnection(masks):  # min_y,min_x,max_y,max_x
    # Create a color label image
    mask1 = masks[0]
    color_label_img = np.ones((mask1.shape[0], mask1.shape[1], 3), dtype=np.uint8)
    for i, mask_data in enumerate(masks):
        # Generate a random color
        color = np.random.randint(0, 256, 3)
        mask = mask_data
        # Apply the color to the mask
        color_label_img[mask] = color_label_img[mask] * color

    # Display the color label image
    plt.imshow(color_label_img)
    plt.show(block=True)
    plt.close()


path2023 = R"C:\Users\14318\scan2bim2024\2d\test\5cm_rooms"
path2024 = "output_mask/"
pathraw = "output_data_2d/"
compare_files = [
    "02_TallOffice_01_F7",
    "02_TallOffice_01_F8",
    "34_Parking_04_F1"
]


def genMaskMap(pointcloud: o3d.geometry.PointCloud):
    global minx, miny, maxx, maxy, xsize, ysize

    mask = np.zeros((xsize, ysize), dtype=bool)
    size = len(pointcloud.points)
    i = 0
    while i < size:
        point = pointcloud.points[i]
        i += 1
        x = int((point[0] - minx) / (maxx - minx) * xsize)
        y = int((point[1] - miny) / (maxy - miny) * ysize)
        if x < 0 or x >= xsize or y < 0 or y >= ysize:
            continue
        mask[x, y] = True
    return mask


for compare_file in compare_files:
    # find dir that contains the file in dir name
    global minx, miny, maxx, maxy, xsize, ysize
    mask2023 = []
    for file in os.listdir(path2024):
        if file.endswith(".npy") and compare_file in file:
            mask2024 = np.load(os.path.join(path2024, file), allow_pickle=True)
            break

    for file in os.listdir(pathraw):
        if file.endswith(".json") and compare_file in file:
            json_file_path = os.path.join(pathraw, file)
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            break
    minx = data["floor"]["min"][0]
    miny = data["floor"]["min"][1]
    maxx = data["floor"]["max"][0]
    maxy = data["floor"]["max"][1]
    xsize = mask2024[0].shape[0]
    ysize = mask2024[0].shape[1]
    print(maxx - minx, maxy - miny, xsize, ysize)
    for dir in os.listdir(path2023):
        if compare_file in dir:
            path = os.path.join(path2023, dir)
            for plyfile in os.listdir(path):
                if plyfile.endswith(".ply"):
                    pointcloud = o3d.io.read_point_cloud(os.path.join(path, plyfile))
                    mask2023.append(genMaskMap(pointcloud))

            break
    confusion_matrix = compute_intersection_matrix(mask2023, mask2024)
    min = np.min(confusion_matrix)
    max = np.max(confusion_matrix)
    print(min, max)
    confusion_matrix = (confusion_matrix - min) / max
    drawconnection(mask2023)
    drawconnection(mask2024)
    # Print the confusion matrix
    plt.imshow(confusion_matrix, cmap='hot', interpolation='nearest')

    plt.show(block=True)
