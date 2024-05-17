from segment_anything import SamPredictor, sam_model_registry
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

sam = sam_model_registry["vit_h"](checkpoint="model_weight/sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)
#using numpy to read image


# Read the image file
img = Image.open('output_data/08_ShortOffice_01_F1_ceiling_low_img.png')

# Resize the image so that the long side is 1024
long_side = max(img.size)
scale_factor = 1024 / long_side
new_size = (round(img.size[0] * scale_factor), round(img.size[1] * scale_factor))
img = img.resize(new_size)
img_rgb = img.convert('RGB')
# Convert the image to a numpy array
img_np = np.array(img_rgb)

# Normalize the image data to 0-1
img_np = img_np / 255.0

# Add an extra dimension for batch size at the beginning
img_np = np.expand_dims(img_np, axis=0)

# Transpose the numpy array to BCHW format
#img_np = np.transpose(img_np, (0, 3, 1, 2))
img_np = img_np.squeeze(0)
print(img_np.shape)
predictor.set_image(img_np)
masks, _, _ = predictor.predict()
masks = masks.squeeze()
masks = np.transpose(masks, (1, 2, 0))
# Convert boolean masks to uint8
masks = masks.astype(np.uint8) * 255

# Save the image
plt.imsave('output_data/seg_room/08_ShortOffice_01_F1_ceiling_low_img_mask.png', masks)

# Display the image
plt.imshow(masks)