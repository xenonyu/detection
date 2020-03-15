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

from utils import display_instances, get_larger_box, exportTxt, exportCondition, videoToPic, handPoseImage, detectFinger, detectCornerYOLO, detectCornerMask

def detectPhone(modelPhone, frameName, PILImage):
    # Run detection
    resultImg, resultBox = modelPhone.detect_image(PILImage)

    # 若成功则输出到文件
    if resultBox.shape[1] != 4:
        print("phone not found on the ", frameName, "frame, process next\n")
        return None, None
    print("成功找到手机")
    # 打印结果个数
    print("手机个数：", resultBox.shape[0])
    
    # 计算手机大小
    phoneSize = (resultBox[0][3] - resultBox[0][1]) * (resultBox[0][2] - resultBox[0][0])
    
    # 得到手机部分放大的框
    phoneBox = get_larger_box(xy = resultBox, scale = 3, w = PILImage.size[0], h = PILImage.size[1])
    phoneBox = phoneBox[0]
    phoneBox = [int(i) for i in phoneBox]
    print(" phone box: ", phoneBox) 
    return phoneBox, phoneSize


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
    frames = videoToPic(videoPath, step = 1, nframe = 20)
    
    condition = 0
    
    for i in range(len(frames) - 1):
        print('processing: {}\n'.format(frames[i][0]))
        frameName = frames[i][0]
        frameName2 = frames[i+1][0]
        CVImage = frames[i][1]
        CVImage2 = frames[i+1][1]
        PILImage = Image.fromarray(cv2.cvtColor(CVImage,cv2.COLOR_BGR2RGB))
        PILImage2 = Image.fromarray(cv2.cvtColor(CVImage2, cv2.COLOR_BGR2RGB))
        print(PILImage.size[0:2])
        print(CVImage.shape)

        phoneBox, phoneSize = detectPhone(modelPhone, frameName, PILImage)
        phoneBox2, phoneSize2 = detectPhone(modelPhone, frameName2, PILImage2)
        if(phoneBox == None or phoneBox2 == None):
            continue
        
        # 得到剪切的手机部分
        croppedCV = CVImage[phoneBox[1] : phoneBox[3], phoneBox[0] : phoneBox[2]]
        croppedPIL = Image.fromarray(cv2.cvtColor(croppedCV,cv2.COLOR_BGR2RGB))  
        croppedCV2 = CVImage[phoneBox2[1] : phoneBox2[3], phoneBox2[0] : phoneBox2[2]]
        croppedPIL2 = Image.fromarray(cv2.cvtColor(croppedCV2,cv2.COLOR_BGR2RGB))
        
        # 将截取的部分输入handposeimage识别指尖
        finger_box = detectFinger(image = [croppedCV, croppedCV2], name = frameName, phoneBox = phoneBox)
        print("finger box: {}".format(finger_box))
        if(finger_box is None):
            continue
        if(condition < 2):
            condition = 2
        
        # 将截取的部分输入mask-rcnn corner_phone模型识别手机角
        corner_box = detectCornerYOLO(ModelCornerP = modelCornerP, image = croppedPIL, name = frameName, phoneBox = phoneBox)
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
       