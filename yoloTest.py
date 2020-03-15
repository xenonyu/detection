import os
import cv2
import pdb
import sys, getopt
import numpy as np
from PIL import Image
from datetime import datetime

sys.path.append('/backup/xymmodel/yolo/')
sys.path.append('/backup/xymmodel/yoloCorner')
sys.path.append('/home/user/zy/attack-on-pattern-pin/detection/util/')
from yolo import YOLO
from yoloCorner import YOLOCorner

from detection import ModelCornerPhone, delete_redu

from utils import display_instances, get_larger_box, exportTxt, exportCondition, videoToPic, handPoseImage

def detectFinger(image, phoneBox, name):
    fPosition = [0,0,0]
    fPosition = handPoseImage(image)

    #判断是否成功
    if(fPosition != None):
        #计算原始坐标 finger Position[x0, y0, type], phone_box[x0, y0, x0, y0]
        left = fPosition[0] + phoneBox[0] -13
        up = fPosition[1] + phoneBox[1] - 13
        right = fPosition[0] + phoneBox[0] + 13
        bottom = fPosition[1] + phoneBox[1] + 13
        finger_box = [left, up, right, bottom]
        return np.array(finger_box)
    else:
        print("handpose fail on the " , name, "frame, process next\n")
        return(None)
    
def detectCornerYOLO(ModelCornerP, image, phoneBox, name):
    # Run detection
    __, result_box = modelCornerP.detect_image(image)

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

def loadModel():
    #加载模型
    yolo = YOLO()
    modelC = YOLOCorner()
    # modelCornerPhone = ModelCornerPhone()
    # modelC = modelCornerPhone.getModelCornerPhone()
    return yolo, modelC

def batchProcess(videoDir, outputPath, modelPhone, modelCornerP):
    for dir, __, fileNames in os.walk(videoDir):
        fileNames.sort(key = lambda x: int(x[-9:-4]))
        for fileName in fileNames:
            videoPath = os.path.join(dir, fileName)
            print(videoPath)
            process(videoPath = videoPath, outputPath = outputPath, modelPhone = modelPhone, modelCornerP = modelCornerP)

def process(videoPath, outputPath, modelPhone, modelCornerP):

    videoName = str(videoPath).split("/")[-1].split(".")[0]

    #切帧
    videoToPic(videoPath, step = 2)

    #加载图片名
    frameNames = os.listdir(os.path.join(IMAGE_DIR, videoName))
    frameNames.sort(key = lambda x: int(x[:-4])) #
    
    condition = 0
    
    for frameName in frameNames:
        print('processing: {}\n'.format(frameName))

        PILImage = Image.open(os.path.join(IMAGE_DIR, videoName, frameName))
        CVImage = cv2.imread(os.path.join(IMAGE_DIR, videoName, frameName))
        # print(PILImage.size[0:2])
        # print(CVImage.shape)
        # break

        # Run detection
        result_img, result_box = modelPhone.detect_image(PILImage)

        # 若成功则输出到文件
        if result_box.shape[1] != 4:
            print("phone not found on the ", frameName, "frame, process next\n")
            continue
        if condition < 1:
            condition = 1
        print("成功找到手机")
        # 打印结果个数
        print("手机个数：", result_box.shape[0])
        
        # 计算手机大小
        phoneSize = (result_box[0][3] - result_box[0][1]) * (result_box[0][2] - result_box[0][0])
        
        # 得到手机部分放大的框
        phone_box = get_larger_box(xy = result_box, scale = 3, w = PILImage.size[0], h = PILImage.size[1])
        phone_box = phone_box[0]
        phone_box = [int(i) for i in phone_box]
        print(" phone box: ", phone_box)
        
        # 得到剪切的手机部分
        croppedCV = CVImage[phone_box[1] : phone_box[3], phone_box[0] : phone_box[2]]
        croppedPIL = Image.fromarray(cv2.cvtColor(croppedCV,cv2.COLOR_BGR2RGB))  
        
        # 将截取的部分输入handposeimage识别指尖
        finger_box = detectFinger(image = croppedCV, name = frameName, phoneBox = phone_box)
        if(finger_box is None):
            continue
        if(condition < 2):
            condition = 2
        
        # 将截取的部分输入mask-rcnn corner_phone模型识别手机角
        corner_box = detectCornerYOLO(ModelCornerP = modelCornerP, image = croppedPIL, name = frameName, phoneBox = phone_box)
        if(corner_box is not None):
            condition = 3
            
        # 显示图像
        # visualize.display_instances(frameName, os.path.join(outputPath, videoName), CVImage, r_phone['rois'], r_phone['masks'], r_phone['class_ids'],
        #                                 class_names, r_phone['scores'])

        # 将手指框画上
        draw = cv2.rectangle(CVImage, (finger_box[0], finger_box[1]), (finger_box[2], finger_box[3]), (0,255,0), 2)
        # 将角框画上
        if(corner_box is not None):
            draw = cv2.rectangle(CVImage, (corner_box[0],corner_box[1]), (corner_box[2],corner_box[3]), (0,255,0), 2)

        #创建文件夹
        if not os.path.exists(os.path.join(outputPath, videoName)):
            os.makedirs(os.path.join(outputPath, videoName))
        
        #输出识别的图片
        cv2.imwrite(os.path.join(outputPath, videoName, "detection.jpg"), draw)

        #输出txt
        exportTxt(frameName = frameName, videoName = videoName, cornerBox = corner_box, fingerBox = finger_box, outputPath = outputPath, phoneSize = phoneSize)
        
        break
    exportCondition(videoName, condition)  

if __name__ == '__main__':
    # Root directory of the project
    ROOT_DIR = "/home/user/zy/attack-on-pattern-pin/"

    # Directory of images to run detection on
    IMAGE_DIR = os.path.join(ROOT_DIR, "detection/input_images")
    OUTPUT_PATH = os.path.join(ROOT_DIR, "results")
    
    #get param
    try:
      opts, args = getopt.getopt(sys.argv[1:],"hd:p:o:",["videoDir=","videoPth=", "outputPth="])
    except getopt.GetoptError:
        print( 'yoloTest.py --videoDir <video dir> --videoPth <video path> \
            \n or \n yoloTest.py -d <video dir> -p <video path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(( 'usage: \n yoloTest.py --videoDir <video dir> --videoPth <video path> \
            \n or \n yoloTest.py -d <video dir> -p <video path>'))
            sys.exit()
        elif opt in ("-d", "--videoDir"):
            option = "DIR"
            VIDEO_DIR = os.path.abspath(arg)
        elif opt in ("-p", "--videoPth"):
            option = "PTH"
            VIDEO_PATH = os.path.abspath(arg) 
        elif opt in ("-o", "--outputPth"):
            OUTPUT_PATH = os.path.abspath(arg)
    
    modelPhone, modelCornerP = loadModel() 
    if option == "DIR":
        batchProcess(VIDEO_DIR, OUTPUT_PATH, modelPhone = modelPhone, modelCornerP = modelCornerP)
    elif option == "PTH":
        process(VIDEO_PATH, OUTPUT_PATH, modelPhone = modelPhone, modelCornerP = modelCornerP)
       