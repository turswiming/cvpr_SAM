import os

import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy import ndarray
from collections import defaultdict
import shutil
import json


def split_masks(mask):
    num_labels, labels = cv2.connectedComponents(mask.astype('uint8'))
    masks = []
    for label in range(1, num_labels):
        component_mask = (labels == label).astype(bool)
        masks.append(component_mask)
    return masks


def compare_mask_data(mask1):
    return -mask1["area"]


def getName(path: str) -> str:
    name = os.path.basename(path)
    names = name.split('_')[0:4]
    return '_'.join(names)


def calculatebbox(mask: np.ndarray) -> tuple[int, int, int, int]:  # minx,maxx,miny,maxy
    # calculate the bounding box of the mask
    indices = np.where(mask >= 1)
    maxy = np.max(indices[0])
    maxx = np.max(indices[1])
    miny = np.min(indices[0])
    minx = np.min(indices[1])

    return miny, minx, maxy, maxx


def plot_stability_scores(scores):
    plt.hist(scores, bins='auto')
    plt.title('Stability Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.show()


def bboxcollide(bbox1: tuple[int, int, int, int], bbox2: tuple[int, int, int, int]) -> bool:  # min_y,min_x,max_y,max_x
    min_y1, min_x1, max_y1, max_x1 = bbox1
    min_y2, min_x2, max_y2, max_x2 = bbox2

    return not (min_y1 > max_y2 or min_y2 > max_y1 or min_x1 > max_x2 or min_x2 > max_x1)


def occlusion_percentage(mask: np.ndarray,
                         ceiling_map: np.ndarray,
                         ) -> float:  # min_y,min_x,max_y,max_x
    num_samples = 300
    true_indices = np.argwhere(mask)
    sampled_indices = np.random.choice(true_indices.shape[0], min(num_samples, true_indices.shape[0]), replace=False)
    ceiling_size = 0
    for i in sampled_indices:
        index = true_indices[i]
        if ceiling_map[
            int(index[0] / mask.shape[0] * ceiling_map.shape[0]),
            int(index[1] / mask.shape[1] * ceiling_map.shape[1])
        ] == 0:
            ceiling_size += 1
    return ceiling_size / num_samples


def drawconnection(connection: list[tuple[int, int]],
                   bbox: dict[int, tuple[int, int, int, int]],
                   masks: list[np.ndarray],
                   text: str):  # min_y,min_x,max_y,max_x
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
    for i in connection:
        miny1, minx1, maxy1, maxx1 = bbox[i[0]]
        miny2, minx2, maxy2, maxx2 = bbox[i[1]]
        color = np.random.rand(3)

        plt.plot([minx1, maxx1, maxx1, minx1, minx1], [miny1, miny1, maxy1, maxy1, miny1], color=color)
        plt.plot([minx2, maxx2, maxx2, minx2, minx2], [miny2, miny2, maxy2, maxy2, miny2], color=color)

        # connect the center of the two masks
        plt.plot([(minx1 + maxx1) / 2, (minx2 + maxx2) / 2], [(miny1 + maxy1) / 2, (miny2 + maxy2) / 2], color=color)
    plt.text(1, 3, text)

    plt.show(block=True)

    # Display text at the coordinates (1, 3)

    plt.show()
    plt.close()


# def drawconnection(masks):  # min_y,min_x,max_y,max_x
#     mask1 = masks[0]
#     color_label_img = np.ones((mask1.shape[0], mask1.shape[1], 3), dtype=np.uint8)
#     for i, mask_data in enumerate(masks):
#         color = np.random.randint(0, 256, 3)
#         mask = mask_data
#         color_label_img[mask] = color_label_img[mask] * color
#
#     plt.imshow(color_label_img)
#     plt.show(block=True)
#     plt.close()

def lerp(v0, v1, t):
    return (1 - t) * v0 + t * v1


def realconnection(mask1: np.ndarray, mask2: np.ndarray):
    # Convert masks to uint8
    mask1 = mask1.astype(np.uint8)
    mask2 = mask2.astype(np.uint8)

    # Erode the masks
    kernel = np.ones((5, 5), np.uint8)
    mask1 = cv2.dilate(mask1, kernel, iterations=1)
    mask2 = cv2.dilate(mask2, kernel, iterations=1)

    intersection = cv2.bitwise_and(mask1, mask2)

    # Check if there is any intersection
    if np.any(intersection > 0):
        return True
    else:
        return False


def connectionsize(mask1: np.ndarray, mask2: np.ndarray) -> int:
    # Convert masks to uint8
    mask1 = mask1.astype(np.uint8)
    mask2 = mask2.astype(np.uint8)

    # Erode the masks
    kernel = np.ones((5, 5), np.uint8)
    mask1 = cv2.dilate(mask1, kernel, iterations=1)
    mask2 = cv2.dilate(mask2, kernel, iterations=1)

    intersection = cv2.bitwise_and(mask1, mask2)

    return np.sum(intersection)


def merge_long_connection(
        connection: list[tuple[int, int]],
        masks: np.ndarray,
        bbox: dict[tuple[int, int, int, int]]) -> list[np.ndarray]:
    mask_sizes = {}
    for i in range(len(connection)):
        mask1 = masks[connection[i][0]]
        mask2 = masks[connection[i][1]]
        connect_size = connectionsize(mask1, mask2)
        if mask_sizes.get(connection[i][0]) is None:
            mask_sizes[connection[i][0]] = np.sum(mask1)
        if mask_sizes.get(connection[i][1]) is None:
            mask_sizes[connection[i][1]] = np.sum(mask2)
        if connect_size > 0.6 * mask_sizes[connection[i][0]] or connect_size > 0.6 * mask_sizes[connection[i][1]]:
            masks[connection[i][0]] = masks[connection[i][0]] + masks[connection[i][1]]
            masks[connection[i][1]] = np.zeros(masks[connection[i][1]].shape)
    new_masks = []
    for i in range(len(masks)):
        if np.sum(masks[i]) > 0:
            new_masks.append(masks[i])
    return new_masks
    pass


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    plt.show()


def isRemovable(mask: np.ndarray) -> bool:
    indices = np.where(mask >= 1)
    if len(indices[0]) < 500:
        return True

    if (indices[0].max() - indices[0].min()) < 50 or (indices[1].max() - indices[1].min()) < 50:
        return True
    dilated_mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    dilated_mask_indices = np.where((dilated_mask >= 1) & (mask == 0))
    if len(dilated_mask_indices[0]) * 7 > len(indices[0]):
        return True
    return False


def cut_small(connection: list[tuple[int, int]], masks: list[np.ndarray]) \
        -> list[np.ndarray]:
    remove_mask_indices = []
    for i in range(len(masks)):
        if isRemovable(masks[i]):
            remove_mask_indices.append(i)
            connects = []
            for c in connection:
                if c[0] == i:
                    connects.append(c[1])
                if c[1] == i:
                    connects.append(c[0])
            if len(connects) == 1:
                masks[connects[0]] = masks[connects[0]] + masks[i]
            else:
                max_id = -1
                maxvalue = -1
                for maskID in connects:
                    val = connectionsize(masks[maskID], masks[i])
                    if val > maxvalue:
                        maxvalue = val
                        max_id = maskID
                if max_id != -1:
                    masks[max_id] = masks[max_id] + masks[i]

    new_masks = []
    for i in range(len(masks)):
        if i not in remove_mask_indices:
            new_masks.append(masks[i])

    return new_masks


def cut_single(connection: list[tuple[int, int]], masks: list[np.ndarray]) \
        -> list[np.ndarray]:
    remove_mask_indices = []
    single_connection_mask = []
    for i in range(len(masks)):
        connects = []
        for c in connection:
            if c[0] == i:
                connects.append(c[1])
            if c[1] == i:
                connects.append(c[0])
        if len(connects) == 1:
            single_connection_mask.append(i)
    for a in range(len(single_connection_mask)):
        i = single_connection_mask[a]
        remove_mask_indices.append(i)
        connects = []
        for c in connection:
            if c[0] == i:
                connects.append(c[1])
            if c[1] == i:
                connects.append(c[0])
        if len(connects) == 1:
            masks[connects[0]] = masks[connects[0]] + masks[i]
        else:
            max_id = -1
            maxvalue = -1
            for maskID in connects:
                val = connectionsize(masks[maskID], masks[i])
                if val > maxvalue:
                    maxvalue = val
                    max_id = maskID
            if max_id != -1:
                masks[max_id] = masks[max_id] + masks[i]

    new_masks = []
    for i in range(len(masks)):
        if i not in remove_mask_indices:
            new_masks.append(masks[i])

    return new_masks


def genConnection(masks: list[np.ndarray]) -> tuple[list[tuple[int, int]], dict[int, tuple[int, int, int, int]]]:
    potential_connection = []
    bbox = {}
    for i in range(len(masks)):
        j = i + 1
        while j < len(masks):
            if bbox.get(i) is None:
                bbox[i] = calculatebbox(masks[i])
            if bbox.get(j) is None:
                bbox[j] = calculatebbox(masks[j])  # shape[0]:y, shape[1]:x
            if bboxcollide(bbox[i], bbox[j]):  # min_y,min_x,max_y,max_x
                potential_connection.append((i, j))
            j += 1
    true_connection: list[tuple[int, int]] = []
    for i in potential_connection:
        a = masks[i[0]]
        b = masks[i[1]]
        if realconnection(a, b):
            true_connection.append(i)
    return true_connection, bbox


def split_stability_score_map(stability_score_map):
    # Get all unique scores
    unique_scores = np.unique(stability_score_map)

    # Create a mask for each unique score
    masks = []
    for score in unique_scores:
        mask = np.where(stability_score_map == score, True, False)
        masks.append(mask)

    return masks


def dfs(node, graph, visited):
    visited.add(node)
    size = 1
    for neighbor in graph[node]:
        if neighbor not in visited:
            size += dfs(neighbor, graph, visited)
    return size


def largest_connected_component(connection):
    # Convert the connection list to a graph
    graph = defaultdict(list)
    for node1, node2 in connection:
        graph[node1].append(node2)
        graph[node2].append(node1)

    max_size = 0
    max_component = None
    for node in graph:
        visited = set()
        if node not in visited:
            size = dfs(node, graph, visited)
            if size > max_size:
                max_size = size
                max_component = visited.copy()
    return max_component


# npy is short for numpy file, don`t understand it in chinese pinyin.
def processnpy(masks: np.ndarray, name: str) -> tuple[list[ndarray | list[ndarray]] | list[ndarray], dict[
    str, int]] | None:
    # step 1: load the ceiling_high_mask for denoise -------------------------------------------------------------------
    # dict_keys(['segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'])
    ceiling_high_map_mask = None
    for file in os.listdir('output/output_data_2d/'):
        if file.endswith('ceiling_mask.png'):
            if name in file:
                ceiling_high_map_mask = cv2.imread('output/output_data_2d/' + file, cv2.IMREAD_UNCHANGED)
                break
    if ceiling_high_map_mask is None:
        print("ceiling_high_map_mask for {} not found".format(name))
        return
    # step 2: split the masks, remove overlap --------------------------------------------------------------------------
    new_masks = []
    # sort mask by area, see function compare_mask_data for details
    masks = sorted(masks, key=compare_mask_data)
    firstMask = masks[0]["segmentation"]
    stability_score_map = np.zeros((firstMask.shape[0], firstMask.shape[1]), dtype=np.float32)
    # id = 2 to skip background
    id = 2
    for mask_info in masks:
        mask = mask_info["segmentation"]
        stability_score_map[mask] = np.maximum(stability_score_map[mask], id)
        id += 1
    mask_cuted = split_stability_score_map(stability_score_map)

    for mask in mask_cuted:
        new_masks.extend(split_masks(mask))

    pass
    # step 3: merge small masks to connected larger masks --------------------------------------------------------------
    connection, bbox = genConnection(new_masks)
    drawconnection(connection, bbox, new_masks, name)

    masks = cut_small(connection, new_masks)
    connection, bbox = genConnection(masks)
    drawconnection(connection, bbox, masks, name)

    pass
    # step 4: remove masks that are occlusion, based on occlusion map --------------------------------------------------
    occlusion_percentages = []
    kernel = np.ones((35, 35), np.uint8)
    ceiling_high_map = cv2.dilate(ceiling_high_map_mask.astype(np.uint8), kernel, iterations=1)

    all_percent = occlusion_percentage(np.ones(masks[0].shape, bool), ceiling_high_map)
    room_masks = []
    if all_percent >= 0.8:
        # step 4.1: if occlusion map are too empty, just remove masks connected to corner ------------------------------
        for i in range(len(masks)):
            if masks[i][0, 0] == 0 and masks[i][0, -1] == 0 and masks[i][-1, 0] == 0 and masks[i][-1, -1] == 0:
                room_masks.append(masks[i])
        pass
    else:
        # step 4.2: check if a mask is an occlusion mask based on occlusion map ----------------------------------------
        for mask in masks:
            percentage = occlusion_percentage(mask, ceiling_high_map)
            occlusion_percentages.append(percentage)
        # use kmeans to cluster the occlusion percentage
        print(occlusion_percentages)
        arr = np.array(occlusion_percentages)
        size = len(arr)
        # Find the max and min density
        if size > 10:
            max_indices = np.argpartition(arr, -int(size * 0.3 + 1))[-int(size * 0.3 + 1):]
            min_indices = np.argpartition(arr, 2)[:2]
        else:
            max_indices = np.argpartition(arr, -1)[-1:]
            min_indices = np.argpartition(arr, 1)[:1]
        max_percentage = sum(arr[max_indices]) / len(max_indices)
        min_percentage = sum(arr[min_indices]) / len(min_indices)
        print("max" + str(max_percentage))
        print("min" + str(min_percentage))
        threshold = lerp(all_percent, 1, 0.1)
        print("threshold" + str(threshold))
        for i in range(len(occlusion_percentages)):
            if occlusion_percentages[i] < threshold:
                room_masks.append(masks[i])
        pass

    pass
    # step 5: remove noise by large connected subgraph -----------------------------------------------------------------
    print(len(room_masks))
    room_masks_dilated = []
    for room_mask in room_masks:
        kernel = np.ones((13, 13), np.uint8)
        room_mask_dilated = cv2.erode(room_mask.astype(np.uint8), kernel, iterations=1)

        room_mask_dilated = cv2.dilate(room_mask_dilated.astype(np.uint8), kernel, iterations=1)
        room_masks_dilated.append(room_mask_dilated)
    connection, bbox = genConnection(room_masks)
    largest_component = largest_connected_component(connection)
    if largest_component is not None:
        room_masks = [room_masks[i] for i in largest_component]
    connection, bbox = genConnection(room_masks)
    if len(connection) > 0:
        drawconnection(connection, bbox, room_masks, name)

    pass
    # step 6: save json file for analysis ------------------------------------------------------------------------------
    room_sizes = [np.sum(mask) for mask in room_masks]
    # get room sizes greater than 1000
    large_room = [room_masks[i] for i in range(len(room_masks)) if room_sizes[i] > 100 * (1 / 0.05) * (1 / 0.05)]
    small_room = [room_masks[i] for i in range(len(room_masks)) if room_sizes[i] <= 10 * (1 / 0.05) * (1 / 0.05)]
    print("finish")
    room_data = {
        "room_number": len(room_masks),
        "large_room": len(large_room),
        "small_room": len(small_room),
    }

    return room_masks, room_data


if __name__ == '__main__':
    path = 'output/output_data_2d_17/'
    save_path_prefix = "output/output_mask_2d_"
    maxnumber = 0
    for file in os.listdir("./output/"):
        dir_prefix = save_path_prefix.split("/")[-1]
        if file.startswith(dir_prefix):
            number = int(file.removeprefix(dir_prefix))
            if number > maxnumber:
                maxnumber = number

    os.mkdir(save_path_prefix + str(maxnumber + 1))
    save_path = save_path_prefix + str(maxnumber + 1) + "/"
    # copy this python file to save path

    for file in os.listdir(path):
        if file.endswith('.npy'):
            if "_ceiling_" in file:
                npy = np.load(path + file, allow_pickle=True)
                print("Processing", file)
                name = getName(file)

                room_masks, room_data = processnpy(npy, name)
                np.save(save_path + name + '_room_mask.npy', room_masks)

                with open(save_path + "{}_room_data.json".format(name), 'w') as f:
                    json.dump(room_data, f)

    shutil.copy("_02_segroom.py", save_path)
    shutil.copy("_03_postprocess.py", save_path)
