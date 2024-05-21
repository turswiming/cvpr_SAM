import os
import numpy as np
from PIL import Image
import cv2


def getBaseNameWithoutExtension(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def selectFile(prefix: str, end: str,path) -> str:
    for file in os.listdir(path):
        fileName = getBaseNameWithoutExtension(file)
        if fileName.startswith(prefix) and fileName.endswith(end):
            return file
    return None


def normalizeImage(img: np.ndarray,scale: float = 1) -> np.ndarray:
    img = img.astype(np.float64)  # Convert img to float64
    img *= scale
    img = img.clip(0, 255)
    img -= img.min()
    img /= img.max()
    img *= 255
    return img.astype(np.uint8)  # Convert img back to uint8

path = "output_data"
names = []
for file in os.listdir(path):
    
    if file.endswith("_bbox.json"):
        names.append(file.removesuffix("_bbox.json"))
    
set_names = set(names)
names = list(set_names)
for name in names:
    
    ceiling_high = name+ "_ceiling_high_img.png"
    ceiling_low = name+ "_ceiling_low_img.png"
    floor_high = name+ "_floor_high_img.png"
    if ceiling_high is None:
        print(path+"/"+name+"ceiling_high not found")
    if ceiling_low is None:
        print(path+"/"+name+"ceiling_low not found")
    if floor_high is None:
        print(path+"/"+name+"floor_high not found")
        
    img1 = cv2.imread(path+"/"+ceiling_high, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path+"/"+ceiling_low, cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(path+"/"+floor_high, cv2.IMREAD_GRAYSCALE)

    #these magic numbers are the scale factors, come from chromatology theory
    img1 = normalizeImage(img1,7) # A factor
    img2 = normalizeImage(img2,7) # B factor
    img3 = normalizeImage(img3,0.1) # L factor

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    img3 = cv2.resize(img3, (img1.shape[1], img1.shape[0]))
    
    
    rgb_img = cv2.merge((img3, img1, img2))
    lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_LAB2RGB)
    cv2.imwrite(path+"/"+'{}_lab_image.png'.format(name), lab_img)
    cv2.imwrite(path+"/"+'{}_rgb_image.png'.format(name), rgb_img)
    