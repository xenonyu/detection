import cv2
# import PIL.image as img
import time
import sys
import os
import math
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
    frames = []
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
            # cv2.imwrite(filename, frame)
            frames.append((str(i) + ".png", frame))
            
        time.sleep(0.01)
        i = i + 1
        
    print("get frame done!\n")
    return frames
    
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
        
def get_larger_boxs(xys, scale, w, h):

    for i in range(len(xys)):
        xys[i] = get_larger_box(xys[i], scale, w, h)
    return(xys)
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
    # 将直接放大scale倍改为仅放大小边长度
    min = np.array([cx_cy[:, 3][i] if(cx_cy[:, 3][i] < cx_cy[:, 2][i]) else cx_cy[:, 2][i] for i in range(len(cx_cy[:, 3]))], dtype=np.int16)
    left_y = cx_cy[:, 1] - min * 0.5
    left_x = cx_cy[:, 0] - 0.5 * min
    right_y = cx_cy[:, 1] + 0.5 * min
    right_x = cx_cy[:, 0] + 0.5 * min
    left_y = np.array([item if item > 0 else 0 for item in left_y], dtype=np.int16)
    left_x = np.array([item if item > 0 else 0 for item in left_x], dtype=np.int16)
    right_y = np.array([item if item <= h else h for item in right_y], dtype=np.int16)
    right_x = np.array([item if item <= w else w for item in right_x], dtype=np.int16)
    return np.stack((left_x, left_y, right_x, right_y), axis = 1) # w, h

# if __name__ == "__main__":
#     a = np.array([[10, 15, 20, 20]], dtype = np.int16)
#     print(get_larger_box(a, 3, 100, 100))

def zeroNear(probMap, maxPoint, threshold):
    zeroWidth = 50
    # width = probMap.shape[1]
    # height = porbMap.shape[0]
    left = maxPoint[1] - zeroWidth // 2
    up = maxPoint[0] - zeroWidth // 2
    right = maxPoint[1] + zeroWidth // 2
    bottom = maxPoint[0] + zeroWidth // 2
    print(probMap[maxPoint[1], maxPoint[0]])
    # sys.exit()
    print(left)
    print(right)
    probMap[left:right, up:bottom].fill(0)
    # print(probMap[up:bottom, left:right])
    return (probMap)

def handPoseImage(image):
    protoFile = "/home/user/zy/attack-on-pattern-pin/detection/hand/pose_deploy.prototxt"
    weightsFile = "/home/user/zy/attack-on-pattern-pin/detection/hand/pose_iter_102000.caffemodel"
    nPoints = 22
    priority = [8, 7, 6, 4, 3, 5, 9, 13, 12, 11, 10, 0, 1 , 2, 14, 15, 16, 17, 18, 19, 20]
    POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    # frame = cv2.imread(imagePath)
    frame = image
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    aspect_ratio = frameWidth/frameHeight

    threshold = 0.3

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
    for i in priority[0:11]:
        
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))
        
         # Find global maxima of the probMap.
        __, firstProb, __, firstPoint = cv2.minMaxLoc(probMap)
        
        secondProbMap = zeroNear(probMap, firstPoint, threshold)
        __, secondProb, __, secondPoint = cv2.minMaxLoc(secondProbMap)
        # a = input("next")
        
        
        if firstProb > threshold :
            # cv2.circle(frameCopy, (int(firstPoint[0]), int(firstPoint[1])), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            # cv2.putText(frameCopy, "{}".format(i), (int(firstPoint[0]), int(firstPoint[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append([firstPoint[0], firstPoint[1], i, firstProb])
        else :
            points.append([0, 0, 0, 0])      
            
        if( secondProb > threshold):
            # cv2.circle(frameCopy, (int(secondPoint[0]), int(secondPoint[1])), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            # cv2.putText(frameCopy, "{}".format(i), (int(secondPoint[0]), int(secondPoint[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append([secondPoint[0], secondPoint[1], i, secondProb])
        else :
            points.append([0, 0, 0, 0])
    return (points)

def backuphandPoseImage(image):
    protoFile = "/home/user/zy/attack-on-pattern-pin/detection/hand/pose_deploy.prototxt"
    weightsFile = "/home/user/zy/attack-on-pattern-pin/detection/hand/pose_iter_102000.caffemodel"
    nPoints = 22
    priority = [8, 7, 6, 4, 3, 5, 9, 13, 12, 11, 10, 0, 1 , 2, 14, 15, 16, 17, 18, 19, 20]
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
        __, firstProb, __, firstPoint = cv2.minMaxLoc(probMap)
        
        secondProbMap = zeroNear(probMap, firstPoint, threshold)
        __, secondProb, __, secondPoint = cv2.minMaxLoc(secondProbMap)
        # a = input("next")
        
        if( secondProb > threshold):
            cv2.circle(frameCopy, (int(secondPoint[0]), int(secondPoint[1])), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(secondPoint[0]), int(secondPoint[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append([int(secondPoint[0]), int(secondPoint[1]), i])
        else :
            points.append([None])
            
        if firstProb > threshold :
            cv2.circle(frameCopy, (int(firstPoint[0]), int(firstPoint[1])), 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(firstPoint[0]), int(firstPoint[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append([int(firstPoint[0]), int(firstPoint[1]), i ])
        else :
            points.append(None)    

    return (points)


'''
@return
    [22, 6] 
'''
def resetFingerPosition(fPositions, phoneBox):
    rFPositions = np.array(np.zeros(6))
    for fPosition in fPositions:
        if(np.all(fPosition) != 0):
            # change [cx, cy] to [lx, ly, rx, ry] and add reset
            left = fPosition[0] + phoneBox[0] -13
            up = fPosition[1] + phoneBox[1] - 13
            right = fPosition[0] + phoneBox[0] + 13
            bottom = fPosition[1] + phoneBox[1] + 13
            rFPositions = np.vstack((rFPositions, [left, up, right, bottom, fPosition[2], fPosition[3]]))
        else:
            rFPositions = np.vstack((rFPositions, [0, 0, 0, 0, 0, 0]))
    rFPositions = np.delete(rFPositions, 0, axis = 0)
    return (rFPositions)

def detectFinger(image, phoneBox, name):
    fPosition = handPoseImage(image)
    # print(fPosition)
    # print(fPosition[0])
    rFPosition = resetFingerPosition(fPosition, phoneBox)
    
    # print("rFPosition: {}".format(rFPosition))
    # print('shape: {}'.format(np.array(rFPosition).shape))
    return (rFPosition)   
    
def detectCornerYOLO(ModelCornerP, image, phoneBox, name):
    # Run detection
    __, result_box = ModelCornerP.detect_image(image)

    if result_box.shape[1] != 4:
        print("phone not found on the ", name, "frame\n")
        return(None)
    #获取手机角
    corner_box = result_box[0]
    print("原始角框: {}".format(corner_box))
    #corner_box[x0, y0, x1, y1], phoneBox[x0, y0, x1, y1]
    corner_box += [phoneBox[0], phoneBox[1], phoneBox[0], phoneBox[1]]
    #change to [x0, y0, x1, y1]
    print("恢复后角框:{}\n".format(corner_box))
        
    return(corner_box)

def detectCornerMask(modelCornerP, image, phoneBox, name):
    # Run detection
    results_corner_phone = modelCornerP.detect([image], verbose=1)

    #get result
    r_corner_phone = results_corner_phone[0]

    # 删除冗余结果
    delete_redu(r_corner_phone)

    #获取rois
    corner_rois = np.array(r_corner_phone['rois'])

    #判断是否成功
    if corner_rois.shape[0] != 1:
        print("phone corner not found on the", name, "frame\n")
        return(None)
        
    # display_instances(frameName, os.path.join(ROOT_DIR,'detection'), image, r_corner_phone['rois'], r_corner_phone['masks'], r_corner_phone['class_ids'],
    #                             ['BG', 'left_up_corner'], r_corner_phone['scores'])
    
    #获取手机角
    corner_box = corner_rois[0]
    print("原始角框: {}".format(corner_box))
    #corner_box[y0, x0, y1, x1], phoneBox[x0, y0, x1, y1]
    corner_box += [phoneBox[1], phoneBox[0], phoneBox[1], phoneBox[0]]
    #change to [x0, y0, x1, y1]
    corner_box[0], corner_box[1], corner_box[2], corner_box[3] = corner_box[1], corner_box[0], corner_box[3], corner_box[2]
    print("恢复后角框:{}\n".format(corner_box))
        
    return(corner_box)


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

def getMovingFinger(fPosition):
    fPosition = np.array(fPosition)
    maxLen = 0
    result = None
    for i in range(np.size(fPosition, 1)):
        if(fPosition[0][i] != None and fPosition[1][i] != None):
            #计算当前距离
            len = math.pow(fPosition[0][i][1] - fPosition[1][i][1], 2) + math.pow(fPosition[0][i][2] - fPosition[1][i][2], 2)
            if(len > maxLen):
                maxLen = len
                result = fPosition[0][i]
    return result