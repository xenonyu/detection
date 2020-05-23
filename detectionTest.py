import os
import cv2
import pdb
import sys, getopt
import numpy as np
from PIL import Image
from datetime import datetime

sys.path.append('/backup1/xymmodel/yolo/')
sys.path.append('/backup1/xymmodel/yoloCorner')
sys.path.append('/home/user/zy/attack-on-pattern-pin/detection/util/')
from yolo import YOLO, detect_video
from yoloCorner import YOLOCorner

from detection import ModelCornerPhone, delete_redu

from utils import display_instances, get_larger_box, get_larger_boxs, exportTxt, exportCondition, videoToPic, handPoseImage, detectFinger, detectCornerYOLO, detectCornerMask

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
    
    phoneBoxs, frames = detect_video(modelPhone, videoPath, numFrame = 20)
    
    # fingerBoxs shape 22, 20, 4
    if(np.all(phoneBoxs == 0)):
        print("phone not found.")
        exportCondition(videoName, value = 0)
        return
    exportCondition(videoName, value = 1)
    
    phoneBoxs = get_larger_boxs(phoneBoxs, 5, np.array(frames[0]).shape[1], np.array(frames[0]).shape[0])
    # print("phoneBox:\n{}".format(phoneBoxs))
    fingerBoxs = np.zeros((22, 1, 7)).astype(np.int16)
    zeros = np.zeros((22, 1, 7))
    for i in range(len(phoneBoxs)):
        print('processing: {}\n'.format(i))
        if((phoneBoxs[i][0] == [0, 0, 0, 0]).all()):
            fingerBoxs = np.hstack((fingerBoxs, zeros))
            continue
        
        CVImage = frames[i]
        # PILImage = Image.fromarray(cv2.cvtColor(CVImage,cv2.COLOR_BGR2RGB))
        
        # 得到剪切的手机部分
        croppedCV = CVImage[phoneBoxs[i][0][1] : phoneBoxs[i][0][3], phoneBoxs[i][0][0] : phoneBoxs[i][0][2]]
        # croppedPIL = Image.fromarray(cv2.cvtColor(croppedCV,cv2.COLOR_BGR2RGB))
        
        # 将截取的部分输入handposeimage识别指尖
        # cv2.imwrite("/home/user/zy/attack-on-pattern-pin/detection/cropped{}.jpg".format(i), croppedCV)
        
        fingerBox = detectFinger(image = croppedCV, name = str(i), phoneBox = phoneBoxs[i][0])
        # add frameNum
        frameNum = np.empty((22, 1), dtype=np.int16)
        frameNum.fill(i)
        # print(frameNum)
        
        fingerBox = np.hstack((fingerBox, frameNum))
        # print(np.around(fingerBox, decimals = 1))
        fingerBox = fingerBox.reshape(-1, 1, 7)
        # print("finger box: {}".format(fingerBox))
        fingerBoxs = np.hstack((fingerBoxs, fingerBox))
        
    fingerBoxs = np.delete(fingerBoxs, 0, axis = 1)
    print("fingerBoxs: {}.\n phoneBoxs: {}.".format(fingerBoxs[:, 19, :], phoneBoxs))
    
    # fingerBoxs shape 22, 20, 7
    if(np.all(fingerBoxs == 0)):
        print("finger not found.")
        exportCondition(videoName, value = 1)
        return

    exportCondition(videoName, value = 2)
    nframe, fingerBox = getBestFinger(fingerBoxs, option = 0)
    if(nframe is None):
        return
    print("nframe: \n{}\nfingerbox: \n{}".format(nframe, fingerBox))
    #将手指框画上
    draw = cv2.rectangle(frames[nframe], (fingerBox[0], fingerBox[1]), (fingerBox[2], fingerBox[3]), (0,255,0), 2)
    cv2.imwrite(os.path.join("/home/user/zy/attack-on-pattern-pin/detection", videoName + ".jpg"), draw)
    return
    
    logs = "processing video {}...\nget at frame: {}\nfinger box: {}.\n condition: {}.\n\n".format(videoName, nframe, fingerBox, 3)
    print(logs)
    with open(os.path.join("/home/user/zy/attack-on-pattern-pin/detection", "result.txt"), "a+") as f:
        f.write(logs)

    #创建文件夹
    if not os.path.exists(os.path.join(outputPath, videoName)):
        os.makedirs(os.path.join(outputPath, videoName))
    
    #输出识别的图片
    # cv2.imwrite(os.path.join(outputPath, videoName, "detection.jpg"), draw)
   
    # # 输出txt
    # exportTxt(frameName = str(nframe) + ".jpg", videoName = videoName, cornerBox = None, fingerBox = fingerBox, outputPath = outputPath, phoneSize = None)
        
    # exportCondition(videoName, condition)  
    
    
"""  
Args:
    fingerBoxs:
        [22, 20, 7]  指，帧，值  
        值：[left, up, right, bottom, fingerIndex， prob, frameNum]
    option: 
        0: priority to get 8, nature rate
        1: priority to get 8, select moving one
        2: priortiy to get 8, select best prob 
return：

"""    
def getBestFinger(fingerBoxs, option):
    
    fingerBoxs = fingerBoxs.reshape(-1, 7)
    
    priority = [8, 7, 6, 4, 3, 5, 9, 13, 12, 11, 10, 0, 1 , 2, 14, 15, 16, 17, 18, 19, 20]
    if option == 0:
        for i in priority[0:11]:
            print('process finger: {}'.format(i))
            targetFinger = fingerBoxs[fingerBoxs[:, 4] == i, :]
            # print(targetFinger)
            if(targetFinger.size == 0):
                continue
            maxprobIndex = np.argmax(targetFinger[:, 4])
            # print(maxprobIndex)
            if(fingerBoxs[maxprobIndex, 5] != 0):
                print("finger num: {}".format(i))
                return(fingerBoxs[maxprobIndex][6].astype(np.int16), fingerBoxs[maxprobIndex][0:4].astype(np.int16))    
        return(None, None)
    fingerBoxs = fingerBoxs.astype(np.int16)
    avgSpeeds = getAvgSpeeds(fingerBoxs)
    maxArg = np.argmax(avgSpeeds)
    print(avgSpeeds)
    if not avgSpeeds[maxArg]:
        print("no moving fingers found, return casual.")
        
        maxPerFrame = np.max(fingerBoxs[:, :, 1], axis=0)
        print(len(maxPerFrame))
        nframe = -1
        for i in range(len(maxPerFrame)):
            if not maxPerFrame[i]:
                continue
            nframe = i
            trueFinger = np.argmax(fingerBoxs[:, i, 0])
            break
        return (nframe, fingerBoxs[trueFinger, nframe])
    else:
        for j in range(len(fingerBoxs[maxArg, :, 0])):
            if(fingerBoxs[maxArg, j, 0]):
                trueFinger = j
                break
        return(trueFinger, fingerBoxs[maxArg, trueFinger])
    
def getAvgSpeeds(fingerBoxs):
    speed = np.array([0])
    avgSpeeds = np.zeros(22)
    for i in range(fingerBoxs.shape[0]):
        fBArray = fingerBoxs[i]
        print("processing finger {}".format(i))
        print("fBArray: {}".format(fBArray))
        # lst = np.arange(22)
        dic = {}
        last = -1
        speed = np.array([0])
        for j in range(fBArray.shape[0]):
            if not fBArray[j][0]:
                continue
            if last >= 0:
                print(np.linalg.norm(fBArray[j] - fBArray[last]))
                speed = np.append(speed, np.linalg.norm(fBArray[j] - fBArray[last]) / (j - last))
            last = j
            
            if not ((j + 1)<fBArray.shape[0] and fBArray[j + 1][0]):
                continue
            EucliD = np.linalg.norm(fBArray[j] - fBArray[j+1])
            print("EucliD: {}".format(EucliD))
            if EucliD > 8:
                print("exit in: {}".format(j))
                break   
    
        if not ((speed == np.array([0])).all()):
            speed = np.delete(speed, 0, axis = 0)
            avgSpeed = np.mean(speed)
        
        else:
            avgSpeed = np.zeros(1)
        avgSpeeds[i] = avgSpeed
        print("speed: {}".format(speed) )
        print("avgSpeed:{}".format( avgSpeed))
    print("avgSpeeds: {}".format(avgSpeeds))
    return (avgSpeeds)

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
    modelPhone.close_session()
    modelCornerP.close_session()   