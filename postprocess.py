import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import block_reduce

def downscale_max(image, block_size):
    # 如果image是一个二维数组
    if image.ndim == 2:
        small = block_reduce(image, block_size=(block_size, block_size), func=np.max)
    # 如果image是一个三维数组
    elif image.ndim == 3:
        small = block_reduce(image, block_size=(block_size, block_size, 1), func=np.max)
    else:
        raise ValueError("Unsupported image dimensions: %d" % image.ndim)
    return small

def split_masks(mask):
    num_labels, labels = cv2.connectedComponents(mask.astype('uint8'))
    masks = []
    for label in range(1, num_labels):
        component_mask = (labels == label).astype(bool)
        masks.append(component_mask)
    return masks


def compare_mask_data(mask1):
    return mask1["stability_score"]

def getName(path: str)->str:
    name = os.path.basename(path)
    names = name.split('_')[0:4]
    return '_'.join(names)

def calculatebbox(mask:np.ndarray)->tuple[int,int,int,int]:#minx,maxx,miny,maxy
    #calculate the bounding box of the mask
    indices = np.where(mask >=1)
    maxy = np.max(indices[0])
    maxx = np.max(indices[1])
    miny = np.min(indices[0])
    minx = np.min(indices[1])
    
    return miny,minx,maxy,maxx


def plot_stability_scores(scores):
    plt.hist(scores, bins='auto')
    plt.title('Stability Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.show()

def bboxcollide(bbox1:tuple[int,int,int,int],bbox2:tuple[int,int,int,int])->bool: #min_y,min_x,max_y,max_x
    if bbox1[0] < bbox2[2] and bbox1[2] > bbox2[0] and bbox1[1] < bbox2[3] and bbox1[3] > bbox2[1]:
        return True
    return False

def is_occolution(mask:np.ndarray,height_map:np.ndarray)->bool: #min_y,min_x,max_y,max_x
    num_samples=5
    true_indices = np.argwhere(mask)
    sampled_indices = true_indices[np.random.choice(true_indices.shape[0], num_samples, replace=False)]
    occolution_size = 0
    for index in sampled_indices:
        if height_map[
            int(index[0]/mask.shape[0]*height_map.shape[0]),
            int(index[1]/mask.shape[1]*height_map.shape[1])
            ][0] < 0.5:
            occolution_size+=1
    if occolution_size > num_samples/2:
        return True
    return False
    
    

def drawconnection(connection:list[tuple[int,int]],bbox:dict[tuple[int,int,int,int]],masks): #min_y,min_x,max_y,max_x
    # Create a color label image
    mask1 = masks[0]
    color_label_img = np.ones((mask1.shape[0], mask1.shape[1], 3), dtype=np.uint8)
    
    
    
    for i, mask_data in enumerate(masks):
        # Generate a random color
        color = np.random.randint(0, 256, 3)
        mask = mask_data
        # Apply the color to the mask
        color_label_img[mask] = color_label_img[mask]*color

    # Display the color label image
    plt.imshow(color_label_img)    
    for i in connection:
        miny1,minx1,maxy1,maxx1 = bbox[i[0]]
        miny2,minx2,maxy2,maxx2 = bbox[i[1]]
        color = np.random.rand(3)

        plt.plot([minx1,maxx1,maxx1,minx1,minx1],[miny1,miny1,maxy1,maxy1,miny1], color=color)
        plt.plot([minx2,maxx2,maxx2,minx2,minx2],[miny2,miny2,maxy2,maxy2,miny2], color=color)
        
        #connect the center of the two masks
        plt.plot([(minx1+maxx1)/2,(minx2+maxx2)/2],[(miny1+maxy1)/2,(miny2+maxy2)/2], color=color)
    plt.show( block=True)
    
    plt.close()
    

def realconnection(mask1:np.ndarray,mask2:np.ndarray): #min_y,min_x,max_y,max_x
    # Find contours in the masks
    mask1 = mask1.astype(np.uint8)
    mask2 = mask2.astype(np.uint8)
    contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contour from mask1 intersects with any contour from mask2
    for cnt1 in contours1:
        for cnt2 in contours2:
            # If intersection is not empty
            intersection = cv2.intersectConvexConvex(cnt1, cnt2)
            if intersection[1] is not None and intersection[1].any() > 0:
                # Do something
                return True
                break
    return False

def processnpy(masks:np.ndarray,name:str)->np.ndarray:
    #print(masks[0].keys())# dict_keys(['segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'])
    print(len(masks))
    new_masks = []
    score = []
    size = []
    masks = sorted(masks, key=compare_mask_data)
    firstMask = masks[0]["segmentation"]
    stability_score_map = np.zeros((firstMask.shape[0], firstMask.shape[1]), dtype=np.float32)
    #add mask and stability score to the stability_score_map
    for mask_info in masks:
        mask = mask_info["segmentation"]
        stability_score = mask_info["stability_score"]
        stability_score_map[mask] = np.maximum(stability_score_map,stability_score)
    
    for mask in masks:
        score.append(mask["stability_score"])
        size.append(mask["area"])

        mask = mask["segmentation"]
        
        new_masks.extend(split_masks(mask))
    print(len(new_masks))
    new_masks_temp = new_masks
    new_masks =[]
    for mask in new_masks_temp:
        if np.sum(mask) > 1000:
            new_masks.append(mask)
    occolution_filtered_masks = []
    img= cv2.imread('output_data/'+name+'_ceiling_high_img.png')
    ratio = 16
    img = downscale_max(img,ratio)
    for mask in new_masks:
        if not is_occolution(mask,img):
            occolution_filtered_masks.append(mask)
            
    potential_connection = []
    bbox ={}
    for i in range(len(occolution_filtered_masks)):
        j = i+1
        while j < len(occolution_filtered_masks):
            if bbox.get(i) is None:
                bbox[i] = calculatebbox(occolution_filtered_masks[i])
            if bbox.get(j) is None:
                bbox[j] = calculatebbox(occolution_filtered_masks[j])  #shape[0]:y, shape[1]:x
            if bboxcollide(bbox[i],bbox[j]):    #min_y,min_x,max_y,max_x
                potential_connection.append([i,j])
            j += 1
            
    true_connection = []
    for i in potential_connection:
        a = occolution_filtered_masks[i[0]]
        b = occolution_filtered_masks[i[1]]
        if realconnection(a,b):
            true_connection.append(i)
    drawconnection(true_connection,bbox,new_masks)
    #check if a new masks mask a 
    
            
    
if __name__ == '__main__':    
    path = 'output_data/'
    for file in os.listdir(path):
        if file.endswith('.npy'):
            npy = np.load(path + file,allow_pickle=True)
            name = getName(file)
            
            processnpy(npy,name)
