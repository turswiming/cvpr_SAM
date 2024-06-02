import os
import numpy as np
from PIL import Image
import cv2
from segroom import Seg
import json


def getBaseNameWithoutExtension(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def selectFile(prefix: str, end: str, path) -> str:
    for file in os.listdir(path):
        fileName = getBaseNameWithoutExtension(file)
        if fileName.startswith(prefix) and fileName.endswith(end):
            return file
    return None


def normalizeImage(img: np.ndarray, scale: float = 1) -> np.ndarray:
    img = img.astype(np.float64)  # Convert img to float64
    img *= scale
    img = img.clip(0, 255)
    img -= img.min()
    img /= img.max()
    img *= 255
    return img.astype(np.uint8)  # Convert img back to uint8


def combinechannel(name):
    ceiling_high = name + "_ceiling_high_img.png"
    ceiling_low = name + "_ceiling_low_img.png"
    floor_high = name + "_floor_high_img.png"
    if ceiling_high is None:
        print(path + "/" + name + "ceiling_high not found")
    if ceiling_low is None:
        print(path + "/" + name + "ceiling_low not found")
    if floor_high is None:
        print(path + "/" + name + "floor_high not found")

    img1 = cv2.imread(path + "/" + ceiling_high, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path + "/" + ceiling_low, cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(path + "/" + floor_high, cv2.IMREAD_GRAYSCALE)
    json_file_path = path + "/" + name + "_bbox.json"
    with open(json_file_path, "r") as f:
        data = json.load(f)
    ceiling_high = data["ceiling"]["max"][2]
    ceiling_low = data["ceiling"]["min"][2]
    # these magic numbers are the scale factors, come from chronology theory
    img1 = normalizeImage(img1, (ceiling_high - ceiling_low) / (ceiling_high - ceiling_low - 3.15))  # A factor
    img2 = normalizeImage(img2, (ceiling_high - ceiling_low) / (ceiling_high - ceiling_low - 3.15))  # B factor
    img3 = normalizeImage(img3, 0.1)  # L factor

    cv2.imwrite(path + "/temp/" + name + "_ceiling_high_adjustcolor.png", img1)
    cv2.imwrite(path + "/temp/" + name + "_ceiling_low_adjustcolor.png", img2)
    cv2.imwrite(path + "/temp/" + name + "_floor_high_adjustcolor.png", img3)
    seg = Seg("vit_h", 'model_weight/sam_vit_h_4b8939.pth')
    mask1 = seg.segment(path + "/temp/" + name + "_ceiling_high_adjustcolor.png",
                        path + "/" + name + "_ceiling_high_segmented.png", True)
    mask2 = seg.segment(path + "/temp/" + name + "_ceiling_low_adjustcolor.png",
                        path + "/" + name + "_ceiling_low_segmented.png", True)
    mask3 = seg.segment(path + "/temp/" + name + "_floor_high_adjustcolor.png",
                        path + "/" + name + "_floor_high_segmented.png")
    print(len(mask1))
    print(len(mask2))
    print(len(mask3))
    max_len = max(len(mask1), len(mask2), len(mask3))
    min_len = min(len(mask1), len(mask2), len(mask3))

    img1 = (img1 * (min_len / len(mask1))).astype(np.uint8)
    img2 = (img2 * (min_len / len(mask2))).astype(np.uint8)
    img3 = (img3 * (min_len / len(mask3))).astype(np.uint8)

    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    img3 = cv2.resize(img3, (img2.shape[1], img2.shape[0]))

    rgb_img = cv2.merge((img3, img1, img2))
    lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_LAB2RGB)
    cv2.imwrite(path + "/" + '{}_lab_image.png'.format(name), lab_img)
    cv2.imwrite(path + "/" + '{}_rgb_image.png'.format(name), rgb_img)


path = "output/output_data"
names = []
for file in os.listdir(path):

    if file.endswith("_bbox.json"):
        names.append(file.removesuffix("_bbox.json"))

set_names = set(names)
names = list(set_names)
list.sort(names)
for name in names:
    combinechannel(name)
