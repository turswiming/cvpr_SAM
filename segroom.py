from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

sam = sam_model_registry["vit_h"](checkpoint="model_weight/sam_vit_h_4b8939.pth")
sam.to(device='cuda')
predictor = SamPredictor(sam)
#using numpy to read image


# Read the image file
img = Image.open('output_data/08_ShortOffice_01_F1_ceiling_low_img.png')

# Resize the image so that the long side is 1024
long_side = max(img.size)
scale_factor = 1024 / long_side
new_size = (round(img.size[0] * scale_factor), round(img.size[1] * scale_factor))
img = img.resize(new_size)
plt.imsave('output_data/seg_room/resized.png', np.array(img))

img_rgb = img.convert('RGB')
# Convert the image to a numpy array
img_np = np.array(img_rgb)

mask_generator: SamAutomaticMaskGenerator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.92,
    stability_score_thresh=0.95,
    crop_n_layers=0,
    crop_n_points_downscale_factor=1,
    min_mask_region_area=50,  # Requires open-cv to run post-processing
)
masks = mask_generator.generate(img_np)

# Display the image
#plt.imshow(masks)