import cv2
# import PIL.image as img
import time
import sys
import os
import shutil
import pdb

import logging
import random
import itertools
import colorsys

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display
from mrcnn import utils

def videoToPic(videoPath, outputPath = "/home/user/zy/attack-on-pattern-pin/detection/input_images/", 
               nframe = 10, step = 1):
    assert os.path.isfile(videoPath)
    print("\nbegin to get {} frame of {} by step {}.".format(nframe, videoPath, videoPath))
    
    #获取文件名
    videoName = str(videoPath).split("/")[-1].split(".")[0]

    # 获取一个视频打开cap 1 file name
    cap = cv2.VideoCapture(videoPath) 

    #创建文件夹,存在则清空文件夹
    if not os.path.exists(os.path.join(outputPath, videoName)):
        os.makedirs(os.path.join(outputPath, videoName))
    else:
        shutil.rmtree(os.path.join(outputPath, videoName)) # 能删除该文件夹和文件夹下所有文件
        os.mkdir(os.path.join(outputPath, videoName))
    
    # 判断是否打开 
    is_opened = cap.isOpened  
    
    i = 1
    breakFrame = 2 + (nframe - 1) * step
    while is_opened:
        # 判断结束帧号，到达则停止
        if i == breakFrame:
            break
        
        # 读取每一张 flag frame
        (flag, frame) = cap.read()  
        
        #每step帧保存
        if i % step == 0 and flag:
            filename = os.path.join(outputPath, videoName, (str(i) + '.png'))
            print(filename)
            cv2.imwrite(filename, frame)
            
        time.sleep(0.01)
        i = i + 1
        
    print("get frame done!\n")
    
def bbox_to_cv2(bbox):
    left =bbox[0]
    up = bbox[1]
    width=bbox[2]-bbox[0]
    height=bbox[3]-bbox[1]
    return [left, up, width, height]

def exportCondition(name, value):
    with open(os.path.join("/home/user/zy/attack-on-pattern-pin/detection/condition.txt"), "a") as f:
        f.write("{0} {1}\n".format(name, value))
    
def exportTxt(frameName, videoName, cornerBox, fingerBox, outputPath, phoneSize):
    if(cornerBox is not None):
        cornerBox = bbox_to_cv2(cornerBox)
        cornerBox = [str(i) for i in cornerBox]
        cornerBox = " ".join(cornerBox)
    fingerBox = bbox_to_cv2(fingerBox)
    fingerBox = [str(i) for i in fingerBox]
    fingerBox = " ".join(fingerBox)
    frameNum = frameName[0:-4]
    
    # 创建文件夹
    if not os.path.exists(os.path.join(outputPath, videoName)):
        os.makedirs(os.path.join(outputPath, videoName))

    # 写入数据
    with open(os.path.join(outputPath, videoName, "box.txt"), "w") as f:
        f.write("{0} {1} {2} {3}".format(frameNum, phoneSize, cornerBox, fingerBox))
        
def get_larger_box(xy, scale, w, h):
    """
    Args:
        xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
        scale: the scale of box
    Returns: 
        bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    cx_cy = np.concatenate(((xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      (xy[:, 2:] - xy[:, :2]) * scale), axis = 1) # w, h
    
    left_y = cx_cy[:, 1] - 0.5 * cx_cy[:, 3]
    left_x = cx_cy[:, 0] - 0.5 * cx_cy[:, 2]
    right_y = cx_cy[:, 1] + 0.5 * cx_cy[:, 3]
    right_x = cx_cy[:, 0] + 0.5 * cx_cy[:, 2]
    left_y = [item if item > 0 else 0 for item in left_y]
    left_x = [item if item > 0 else 0 for item in left_x]
    right_y = [item if item <= h else h for item in right_y]
    right_x = [item if item <= w else w for item in right_x]
    return np.stack((left_x, left_y, right_x, right_y), axis = 1) # w, h
print(get_larger_box(xy = np.array([[5, 5, 20, 40]]), scale = 3, w = 100, h = 100))

def handPoseImage(image):
    protoFile = "/home/user/zy/attack-on-pattern-pin/detection/hand/pose_deploy.prototxt"
    weightsFile = "/home/user/zy/attack-on-pattern-pin/detection/hand/pose_iter_102000.caffemodel"
    nPoints = 22
    POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    # frame = cv2.imread(imagePath)
    frame = image
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    aspect_ratio = frameWidth/frameHeight

    threshold = 0.1

    t = time.time()
    # input image dimensions for the network
    inHeight = 368
    inWidth = int(((aspect_ratio*inHeight)*8)//8)
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()
    print("time taken by handpose : {:.3f}".format(time.time() - t))

    # Empty list to store the detected keypoints
    points = []

    # i = 4 大拇指尖
    # i = 8 食指尖
    # i = 9 食指中
    # i = 10 食指后
    for i in range(nPoints):
        
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > threshold :   
            cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA) 
            # my code
            if i in [4, 8, 9, 10]:
                point = list(point)
                point.append(i)
                print ("找到第{}个指头，坐标为{}.".format(i, point))
                cv2.imwrite(os.path.join("/home/user/zy/attack-on-pattern-pin/detection/", "finger-skeleton.jpg"), frameCopy)
                return (point)
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(point[0]), int(point[1])))
        else :
            points.append(None)

    # # Draw Skeleton
    # for pair in POSE_PAIRS:
    #     partA = pair[0]
    #     partB = pair[1]

    #     if points[partA] and points[partB]:
    #         cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
    #         cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    #         cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)


    # cv2.imshow('Output-Keypoints', frameCopy)
    # cv2.imshow('Output-Skeleton', frame)


    cv2.imwrite(os.path.join("/home/user/zy/attack-on-pattern-pin/detection/", 'Output-Keypoints.jpg'), frameCopy)
    # cv2.imwrite('Output-Skeleton.jpg', frame)

    print("Total time taken : {:.3f}".format(time.time() - t))
    return (None)


def display_instances(file, output_path, image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        fig, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            #x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))

    if auto_show:
        ax.axis('off')
        h, w, l = np.shape(image)
        fig.set_size_inches(w/100.0,h/100.0)#���width*height����
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace =0, wspace =0)
        plt.margins(0,0)
        plt.savefig(os.path.join(output_path,file))
        #plt.show()
        plt.close()
        
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image