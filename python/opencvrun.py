"""onnx图片需要调整为[B, C, H, W],且需要归一化
"""

from pathlib import Path

import onnx

import numpy as np
import cv2
import time

import sys
import os
os.chdir(sys.path[0])


SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4


def resize_and_pad(image, new_shape):
    """缩放图片并填充为正方形

    Args:
        image (np.Array): 图片
        new_shape (Tuple): [h, w]

    Returns:
        Tuple: 缩放的图片, 填充的宽, 填充的高
    """
    old_size = image.shape[:2]
    ratio = float(new_shape[-1]/max(old_size)) #fix to accept also rectangular images
    new_size = tuple([int(x*ratio) for x in old_size])
    # 缩放高宽的长边为640
    image = cv2.resize(image, (new_size[1], new_size[0]))
    # 查看高宽距离640的长度
    delta_w = new_shape[1] - new_size[1]
    delta_h = new_shape[0] - new_size[0]
    # 使用灰色填充到640*640的形状
    color = [100, 100, 100]
    img_reized = cv2.copyMakeBorder(image, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT, value=color)

    return img_reized, delta_w ,delta_h


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


def check_onnx(onnx_path):
    # 载入onnx模块
    model_ = onnx.load(onnx_path)
    # print(model_)
    # 检查IR是否良好
    try:
        onnx.checker.check_model(model_)
    except Exception:
        print("Model incorrect")
    else:
        print("Model correct")


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


def post(detections, delta_w ,delta_h, img):
    """后处理

    Args:
        detections (np.Array): 检测到的数据 [25200, 85]
        delta_w (int):  填充的宽
        delta_h (int):  填充的高
        img (np.Array): 原图

    Returns:
        np.Array: 绘制好的图片
    """
    boxes = []
    class_ids = []
    confidences = []
    for prediction in detections:
        confidence = prediction[4].item()
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = prediction[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):
                confidences.append(confidence)
                class_ids.append(class_id)
                # 不是0~1之间的数据
                x, y, w, h = prediction[0].item(), prediction[1].item(), prediction[2].item(), prediction[3].item()
                xmin = x - (w / 2)
                ymin = y - (h / 2)
                box = np.array([xmin, ymin, w, h])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD)

    detections = []
    for i in indexes:
        j = i.item()
        detections.append({"class_index": class_ids[j], "confidence": confidences[j], "box": boxes[j]})


    # Step 9. Print results and save Figure with detections
    for detection in detections:

        box = detection["box"]
        classId = detection["class_index"]
        confidence = detection["confidence"]
        print( f"Bbox {i} Class: {classId} Confidence: {confidence}, Scaled coords: [ cx: {(box[0] + (box[2] / 2)) / img.shape[1]}, cy: {(box[1] + (box[3] / 2)) / img.shape[0]}, w: {box[2]/ img.shape[1]}, h: {box[3] / img.shape[0]} ]" )

        # 还原到原图尺寸
        box[0] = box[0] / ((640-delta_w) / img.shape[1])
        box[2] = box[2] / ((640-delta_w) / img.shape[1])
        box[1] = box[1] / ((640-delta_h) / img.shape[0])
        box[3] = box[3] / ((640-delta_h) / img.shape[0])

        xmax = box[0] + box[2]
        ymax = box[1] + box[3]
        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(xmax), int(ymax)), (0, 255, 0), 3)
        img = cv2.rectangle(img, (int(box[0]), int(box[1]) - 20), (int(xmax), int(box[1])), (0, 255, 0), cv2.FILLED)
        img = cv2.putText(img, str(classId) + " " + "{:.2f}".format(confidence),
                          (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    return img


#--------------------------------#
#   推理
#--------------------------------#
def inference():
    ONNX_PATH = "../weights/yolov5s.onnx"
    IMAGE_PATH = "../images/bus.jpg"

    # 获取图片,扩展的宽高
    img, blob, delta_w ,delta_h = get_image(IMAGE_PATH)

    # 获取模型
    model = get_dnn_model(ONNX_PATH)

    start = time.time()
    # 设置模型输入
    model.setInput(blob)
    # 推理模型结果
    detections = model.forward()
    # print(detections[0].shape)                        # [1, 25200, 85]
    detections = np.squeeze(detections[0])              # [25200, 85]

    # Step 8. Postprocessing including NMS
    img = post(detections, delta_w ,delta_h, img)
    end = time.time()
    print((end - start) * 1000)

    cv2.imwrite("./opencv_det.png", img)


if __name__ == "__main__":
    inference()
