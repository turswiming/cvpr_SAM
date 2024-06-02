from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os
from skimage.transform import rescale
from skimage.filters.rank import minimum
from skimage.morphology import disk, ball
from skimage import img_as_ubyte
import cv2
import numpy as np
import json
import torch
import shutil


def getName(path: str) -> str:
    name = os.path.basename(path)
    name = os.path.splitext(name)
    names = name[0].split('_')[0:4]
    return '_'.join(names)


class Seg:
    def __init__(self, type, model_path):
        self.sam = sam_model_registry[type](checkpoint=model_path)
        self.sam.to(device='cuda')
        self.predictor = SamPredictor(self.sam)

    def min_resize(self, image, new_size):
        # Calculate the scale factors
        y_scale = new_size[0] / image.shape[1]
        x_scale = new_size[1] / image.shape[0]
        image_uint8 = img_as_ubyte(image)

        filtered_image = np.empty_like(image_uint8)

        if len(image_uint8.shape) == 2:
            # The image is grayscale
            filtered_image = minimum(image_uint8, disk(2))
            rescaled = rescale(filtered_image, (x_scale, y_scale), mode='reflect')
        else:
            # The image is color
            for channel in range(image_uint8.shape[2]):
                filtered_image[:, :, channel] = minimum(image_uint8[:, :, channel], disk(2))
            rescaled = rescale(filtered_image, (x_scale, y_scale, 1), mode='reflect')
        # Rescale the image

        # Convert back to uint8
        rescaled = (rescaled * 255).astype(np.uint8)

        return rescaled

    def segment(self, read_path: str, save_path: str, mask_generator: SamAutomaticMaskGenerator) -> list[
        dict[str, any]]:
        img = Image.open(read_path)

        name = getName(read_path)


        img = np.array(img)
        min_val = np.min(img)
        max_val = np.max(img)

        # Rescale the grayscale values to the range 0-255
        img = (img / max_val * 255).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
        img_rgb = cv2.applyColorMap(img, cv2.COLORMAP_HSV)
        img_rgb = np.dstack((img, img, img))
        img_np = img_rgb
        torch.cuda.empty_cache()
        masks = mask_generator.generate(img_np)
        print(len(masks))
        plt.figure(figsize=(20, 20))
        plt.imshow(img_rgb * 0.2)
        show_anns(masks)
        plt.axis('off')
        plt.savefig(save_path + '_segmented.png')
        plt.close()
        print('Segmentation saved to', save_path + '_segmented.png')

        return masks


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


# 现对单个通道进行分解，统计分解数量，越少的权重越高

# Read the image file
path = "output_data_2d"
#back up-------------------
if __name__ == '__main__':
    seg = Seg("vit_h", 'model_weight/sam_vit_h_4b8939.pth')

    path = 'output/output_data/'
    save_path_prefix = "output/output_data_3d_"
    maxnumber = 0
    for file in os.listdir("./output/"):
        dir_prefix = save_path_prefix.split("/")[-1]
        if file.startswith(dir_prefix):
            number = int(file.removeprefix(dir_prefix))
            if number > maxnumber:
                maxnumber = number

    os.mkdir(save_path_prefix + str(maxnumber + 1))
    save_path = save_path_prefix + str(maxnumber + 1) + "/"
    for file in os.listdir(path):
        if file.endswith('_ceiling_high_img.png'):
            img = Image.open(path + file)
            maxsize = max(np.asarray(img).shape)
            mask_generator = SamAutomaticMaskGenerator(
                model=seg.sam,
                points_per_side=int(maxsize / 64),  # another magic number, but it works perfectly
                points_per_batch=15,  # fit 16g vram perfectly
                pred_iou_thresh=0.80,
                stability_score_thresh=0.8,  # origin 0.7
                stability_score_offset=-1,
                box_nms_thresh=0.4,  # origin 0.7

                crop_n_layers=0,
                min_mask_region_area=100,  # Requires open-cv to run post-processing
            )
            res = seg.segment(path + file, save_path + file, mask_generator)
            np.save(save_path + file + '_new{}segmented.npy'.format(len(res)), res)
            print('Segmentation saved to', save_path + file + '_new{}segmented.npy'.format(len(res)))

    shutil.copy("./segroom.py", save_path)
    shutil.copytree(path, save_path + "output_data_2d/")

#back up-------------------