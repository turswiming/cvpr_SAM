from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import os
from skimage.transform import rescale
from skimage.filters.rank import minimum
from skimage.morphology import disk, ball
from skimage import img_as_ubyte

import numpy as np

class Seg:
    def __init__(self, model_path):
        self.sam = sam_model_registry["vit_h"](checkpoint=model_path)
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
            rescaled = rescale(filtered_image, (x_scale,y_scale), mode='reflect')
        else:
            # The image is color
            for channel in range(image_uint8.shape[2]):
                filtered_image[:, :, channel] = minimum(image_uint8[:, :, channel], disk(2))
            rescaled = rescale(filtered_image, (x_scale, y_scale,1), mode='reflect')
        # Rescale the image

        # Convert back to uint8
        rescaled = (rescaled * 255).astype(np.uint8)

        return rescaled
    
    def segment(self, read_path: str, save_path: str,min_resize=False)->list[dict[str, any]]:
        img = Image.open(read_path)
        # Resize the image so that the long side is 1024
        # long_side = max(img.size)
        # scale_factor = 2000 / long_side
        # new_size = (round(img.size[0] * scale_factor), round(img.size[1] * scale_factor))
        # if min_resize:
        #     skimage_image = np.array(img)
        #     skimage_image = self.min_resize(skimage_image, new_size)
        #     img = Image.fromarray(skimage_image.astype(np.uint8))
        # else:
        #     img = img.resize(new_size, Image.BOX)
        # plt.imsave('output_data/seg_room/resized.png', np.array(img))

        img_rgb = img.convert('RGB')
        # Convert the image to a numpy array
        img_np = np.array(img_rgb)
        max_size = max(img_np.shape[0],img_np.shape[1])
        mask_generator: SamAutomaticMaskGenerator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side= max_size/60,# another magic number, but it works perfectly
            points_per_batch= 32, #fit 16g vram perfectly
            pred_iou_thresh= 0.92,
            stability_score_thresh=0.75,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=50,  # Requires open-cv to run post-processing
        )
        masks = mask_generator.generate(img_np)
        print(len(masks))
        print(masks[0].keys())
        plt.figure(figsize=(20,20))
        plt.imshow(img_np*0.2)
        show_anns(masks)
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()
        print('Segmentation saved to', save_path)
        return masks
    
  
    

    

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# Read the image file

if __name__ == '__main__':
    seg = Seg('model_weight/sam_vit_h_4b8939.pth')
    path = 'output_data/'
    for file in os.listdir(path):
        if file.endswith('_lab_image.png') or file.endswith('_rgb_image.png'):
            seg.segment(path + file, 'output_data/' + file + '_segmented.png')
            
#现对单个通道进行分解，统计分解数量，越少的权重越高