"""onnx图片需要调整为[B, C, H, W],且需要归一化
"""

from pathlib import Path

import onnx

import numpy as np
import cv2
import time

from utils import resize_and_pad, check_onnx, post, get_index2label

import sys
import os
os.chdir(sys.path[0])


CONFIDENCE_THRESHOLD = 0.4
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4


def get_image(image_path):
    """获取图像

    Args:
        image_path (str): 图片路径

    Returns:
        Tuple: 原图, 输入的tensor, 填充的宽, 填充的高
    """
    img = cv2.imread(str(Path(image_path)))

    img_reized, delta_w ,delta_h = resize_and_pad(img, (640, 640))

    # [H, W, C] -> [B, C, H, W] & BRG2RGB & 归一化等操作
    # 当同时进行swapRB,scalefactor,mean,size操作时，优先按swapRB交换通道，其次按scalefactor比例缩放，然后按mean求减，最后按size进行resize操作
    blob = cv2.dnn.blobFromImage(img_reized,
                                swapRB=True,                    # 交换 Red 和 Blue 通道, BGR2RGB
                                scalefactor=1.0 / 255,          # 图像各通道数值的缩放比例
                                # mean=[0.485, 0.456, 0.406],   # 用于各通道减去的均值
                                # std=[0.229, 0.224, 0.225],    # 没有std，不要自己加
                                size=(640, 640),                # 图片大小 w,h
                                crop=False,                     # 图像裁剪,默认为False.当值为True时，先按比例缩放，然后从中心裁剪成size尺寸
                                ddepth=cv2.CV_32F               # 数据类型,可选 CV_32F 或者 CV_8U
                                )

    return img, blob, delta_w ,delta_h


def get_dnn_model(onnx_path):
    """获取模型

    Args:
        onnx_path (str): 模型路径

    Returns:
        _type_: _description_
    """
    check_onnx(onnx_path)
    model = cv2.dnn.readNetFromONNX(onnx_path)

    return model


#--------------------------------#
#   推理
#--------------------------------#
def inference():
    ONNX_PATH  = "../weights/yolov5s.onnx"
    IMAGE_PATH = "../images/bus.jpg"
    YAML_PATH  = "../weights/yolov5s.yaml"

    # 获取图片,扩展的宽高
    img, blob, delta_w ,delta_h = get_image(IMAGE_PATH)

    # 获取模型
    model = get_dnn_model(ONNX_PATH)

    # 获取label
    index2label = get_index2label(YAML_PATH)

    start = time.time()
    # 设置模型输入
    model.setInput(blob)
    # 推理模型结果
    detections = model.forward()
    # print(detections[0].shape)                        # [1, 25200, 85]
    detections = np.squeeze(detections[0])              # [25200, 85]

    # Step 8. Postprocessing including NMS
    img = post(detections, delta_w ,delta_h, img, CONFIDENCE_THRESHOLD, SCORE_THRESHOLD, NMS_THRESHOLD, index2label)
    end = time.time()
    print((end - start) * 1000)

    cv2.imwrite("./opencv_det.png", img)


if __name__ == "__main__":
    inference()
