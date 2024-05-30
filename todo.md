添加2d alphashape，通过黑白色数量判断是否应当反转alpha shape内的颜色  
mask to be reversed need to erode to compensatory dilate operated on the original mask


如果不需要添加aplhashape的话，通过形态学操作，提取纯黑背景，所有遮罩减去纯黑背景，注意需要先扩大再减小边缘滤波

