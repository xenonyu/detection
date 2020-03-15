from __future__ import division
import os
import sys
import cv2
import warnings
import random
import skimage.io
import numpy as np
from mrcnn.config import Config
from datetime import datetime
import matplotlib.pyplot as plt
from mrcnn import utils
import mrcnn.model as modellib
warnings.filterwarnings('ignore')


# Root directory of the project
ROOT_DIR = "/home/user/zy/attack-on-pattern-pin/"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Directory to save logs and trained model
MODEL_PHONE_DIR = os.path.join(ROOT_DIR, "detection/model_phone")
MODEL_CORNER_PHONE_DIR = os.path.join(ROOT_DIR, "detection/model_corner_phone")

# Local path to trained weights file
COCO_MODEL_PHONE_PATH = os.path.join(MODEL_PHONE_DIR, "mask_rcnn_shapes_13888.h5")
COCO_MODEL_CORNER_PHONE_PATH = os.path.join(MODEL_CORNER_PHONE_DIR, "mask_rcnn_shapes_corner_phone.h5")

# Judge whether model exist
assert os.path.exists(COCO_MODEL_PHONE_PATH)
assert os.path.exists(COCO_MODEL_CORNER_PHONE_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "detection/input_images")

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
MODEL_PHONE_CLASS_NAMES = ['BG', 'phone', 'finger']
MODEL_CORNER_PHONE_CLASS_NAMES = ['BG', 'left_up_corner']

OUTPUT_PATH = os.path.join(ROOT_DIR, "results")

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 384

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50


# import train_tongue
# class InferenceConfig(coco.CocoConfig):
class ModelPhoneInferenceConfig(ShapesConfig):
    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 2 shapes

    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class ModelCornerPhoneInferenceConfig(ShapesConfig):
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shapes

    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class ModelCornerPhone(object):
    @classmethod
    def __init__(self, **kwargs):
        self.config_corner_phone = ModelCornerPhoneInferenceConfig()
        self.model_corner_phone = modellib.MaskRCNN(mode="inference", model_dir=MODEL_CORNER_PHONE_DIR, config=self.config_corner_phone)
        self.model_corner_phone.load_weights(COCO_MODEL_CORNER_PHONE_PATH, by_name=True)
    def getModelCornerPhone(self):
        return self.model_corner_phone
        
# 得到更大的框
def get_larger_box(r, scale):
    center_y = (r['rois'][0][0] + r['rois'][0][2]) // 2
    center_x = (r['rois'][0][1] + r['rois'][0][3]) // 2
    width_y = (r['rois'][0][2] - r['rois'][0][0]) // 2
    width_x = (r['rois'][0][3] - r['rois'][0][1]) // 2
    left_y = center_y - 3 * width_y
    left_x = center_x - 3 * width_x
    right_y = center_y + 3 * width_y
    right_x = center_x + 3 * width_x
    left_y = left_y if left_y > 0 else 0
    left_x = left_x if left_x > 0 else 0
    right_y = right_y if right_y <= image.shape[0] else image.shape[0]
    right_x = right_x if right_x <= image.shape[1] else image.shape[1]
    return ([left_y, left_x, right_y, right_x])


# 删除冗余结果
def delete_redu(r):
    max_phone_score = 0
    n_instance = r['rois'].shape[0]
    instance = 0
    while instance < n_instance:
        if r['class_ids'][instance] == 1:
            if r['scores'][instance] > max_phone_score:
                max_phone_score = r['scores'][instance]
            else:
                r['rois'] = np.delete(r['rois'], instance, axis=0)
                r['class_ids'] = np.delete(r['class_ids'], instance, axis=0)
                r['scores'] = np.delete(r['scores'], instance, axis=0)
                r['masks'] = np.delete(r['masks'], instance, axis=2)
                instance -= 1
                n_instance -= 1
            instance += 1
        elif r['class_ids'][instance] == 2:
            r['rois'] = np.delete(r['rois'], instance, axis=0)
            r['class_ids'] = np.delete(r['class_ids'], instance, axis=0)
            r['scores'] = np.delete(r['scores'], instance, axis=0)
            r['masks'] = np.delete(r['masks'], instance, axis=2)
            instance -= 1
            n_instance -= 1
            instance += 1
            
def export_txt(filename, videoname, cornerbox, fingerbox):
    #创建文件夹
    if not os.path.exists(os.path.join(OUTPUT_PATH, videoname)):
        os.makedirs(os.path.join(OUTPUT_PATH, videoname))

    #写入帧数
    with open(os.path.join(OUTPUT_PATH, videoname, "box.txt"), "w") as f:
        f.write(filename[0:-4] + " ")
    
    #写入手机角
    cornerleftx = corner_box[1]
    cornerlefty = corner_box[0]
    cornerwidth = corner_box[3] - corner_box[1]
    cornerheigh = corner_box[2] - corner_box[0]
    with open(os.path.join(OUTPUT_PATH, videoname, "box.txt"), "a+") as f:
        f.write("1" + " " + str(cornerleftx) + " " + str(cornerlefty) + " " + str(cornerwidth) + " " + str(cornerheigh) + " ")

    #写入手指
    fingerleftx = finger_box[1]
    fingerlefty = finger_box[0]
    fingerwidth = fingerbox[3] - fingerbox[1]
    fingerheight = fingerwidth
    with open(os.path.join(OUTPUT_PATH, videoname, "box.txt"), "a+") as f:
        f.write("2" + " " + str(fingerleftx) + " " + str(fingerlefty) + " " + str(fingerwidth) + " " + str(fingerheight))



if __name__ == '__main__':

    #get param
    
    VIDEO_PATH = sys.argv[1]
    VIDEO_NAME = str(VIDEO_PATH).split("/")[-1].split(".")[0]

    # caculate accuracy
    succ_count = 0
    rate = 0
    flag = 0

    # box to convert
    box = [0, 0, 0, 0]

    #切帧
    videoToPic.videoToPic(VIDEO_PATH, VIDEO_NAME, step = 1)

    #加载图片名
    file_names = os.listdir(os.path.join(IMAGE_DIR, VIDEO_NAME))
    file_names.sort(key = lambda x: int(x[:-4])) #

    #加载模型
    config_phone = ModelPhoneInferenceConfig()
    config_corner_phone = ModelCornerPhoneInferenceConfig()

    # Create model object in inference mode.
    model_phone = modellib.MaskRCNN(mode="inference", model_dir=MODEL_PHONE_DIR, config=config_phone)
    model_corner_phone = modellib.MaskRCNN(mode="inference", model_dir=MODEL_CORNER_PHONE_DIR, config=config_corner_phone)

    # Load weights trained on MS-COCO
    model_phone.load_weights(COCO_MODEL_PHONE_PATH, by_name=True)
    model_corner_phone.load_weights(COCO_MODEL_CORNER_PHONE_PATH, by_name=True)

    for file in file_names:
        print('processing:  ', file)

        image = cv2.imread(os.path.join(IMAGE_DIR, VIDEO_NAME, file))

        #calculate time
        a = datetime.now()

        # Run detection
        results_phone = model_phone.detect([image], verbose=1)
        
        #get result
        r_phone = results_phone[0]

        #calculate time
        b = datetime.now()
        print("time use", (b - a))

        # 删除冗余结果
        delete_redu(r_phone)

        #获取rois
        phone_rois = np.array(r_phone['rois'])

        # 打印结果个数
        print("手机个数：", phone_rois.shape[0])

        # 若成功则输出到文件
        if phone_rois.shape[0] == 1:

            print("成功找到手机")
            flag = 1

            # 得到手机部分放大的框
            phone_box = []
            phone_box = get_larger_box(r_phone, scale = 3)

            # 得到剪切的手机部分
            cropped = image[phone_box[0] : phone_box[2], phone_box[1] : phone_box[3]]
            cv2.imwrite(os.path.join(ROOT_DIR, 'detection', 'cropped.jpg'), cropped)
            # 将截取的部分输入mask-rcnn corner_phone模型识别手机角
            # Run detection
            results_corner_phone = model_corner_phone.detect([cropped], verbose=1)

            #get result
            r_corner_phone = results_corner_phone[0]

            # 删除冗余结果
            delete_redu(r_corner_phone)

            #获取rois
            corner_rois = np.array(r_corner_phone['rois'])

            #判断是否成功
            if corner_rois.shape[0] == 1:
                visualize.display_instances(file, os.path.join(ROOT_DIR,'detection'), cropped, r_corner_phone['rois'], r_corner_phone['masks'], r_corner_phone['class_ids'],
                                            MODEL_CORNER_PHONE_CLASS_NAMES, r_corner_phone['scores'])
                #获取手机角
                corner_box = corner_rois[0]
                print("原始角框: ",corner_box)
                corner_box[0] = corner_box[0] + phone_box[0]
                corner_box[1] = corner_box[1] + phone_box[1]
                corner_box[2] = corner_box[2] + phone_box[0]
                corner_box[3] = corner_box[3] + phone_box[1]
                print("恢复后角框: ", corner_box)

            else:
                print("phone corner not found on the", file, "frame, process next")
                continue

            #计算原始坐标

            # 将截取的部分输入handposeimage识别指尖
            position = [0,0,0]
            position = handPoseImage(cropped)

            #判断是否成功
            if(position != None):
                #计算原始坐标
                a = position[1] + phone_box[0] - 13
                b = position[0] + phone_box[1] - 13
                c = position[1] + phone_box[0] + 13
                d = position[0] + phone_box[1] + 13
                finger_box = [a, b, c, d]
            else:
                print("handpose fail on the " ,file, "frame, process next")
                continue
                

            # 显示图像
            # image = visualize.display_instances(file, os.path.join(OUTPUT_PATH, VIDEO_NAME), image, r_phone['rois'], r_phone['masks'], r_phone['class_ids'],
            #                                 class_names, r_phone['scores'])
        else:
            print("phone not found on the ", file, "frame, process next")
            continue

        # 将手指框画上
        draw = cv2.rectangle(image, (b,a), (d,c), (0,255,0), 2)
        draw = cv2.rectangle(draw, (corner_box[1],corner_box[0]), (corner_box[3],corner_box[2]), (0,255,0), 2)
        print(finger_box)

        #创建文件夹
        if not os.path.exists(os.path.join(OUTPUT_PATH, VIDEO_NAME)):
            os.makedirs(os.path.join(OUTPUT_PATH, VIDEO_NAME))
        
        #输出识别的图片
        cv2.imwrite(os.path.join(OUTPUT_PATH, VIDEO_NAME, "detection.jpg"), draw)

        #输出txt
        export_txt(videoname = VIDEO_NAME, filename = file, cornerbox = corner_box, fingerbox = finger_box)

        break   



    # 计算成功率
    # rate = round(succ_count / all_count, 3)
    # print("success rate:  ", rate)
