import sys
sys.path.append("../")

import yaml
import cv2
import onnx
import time
import numpy as np
import colorsys
from pathlib import Path


def load_yaml(yaml_path: str) -> dict:
    """通过id找到名称

    Args:
        yaml_path (str): yaml文件路径

    Returns:
        yaml (dict)
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        y = yaml.load(f, Loader=yaml.FullLoader)

    return y


def get_image(image_path: str):
    """获取图像

    Args:
        image_path (str): 图片路径

    Returns:
        Tuple: 原图, 输入的tensor, 填充的宽, 填充的高
    """
    image_bgr = cv2.imread(str(Path(image_path)))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # BGR2RGB
    return image_rgb


def resize_and_pad(image, new_shape):
    """缩放图片并填充为正方形

    Args:
        image (np.Array):      图片
        new_shape (list[int]): [h, w]

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
    image_reized = cv2.copyMakeBorder(image, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT, value=color)

    return image_reized, delta_w ,delta_h


def transform(image: np.ndarray, openvino_preprocess=False) -> np.ndarray:
    """图片预处理

    Args:
        image (np.ndarray): 经过缩放的图片
        openvino_preprocess (bool, optional): 是否使用了openvino的图片预处理. Defaults to False.

    Returns:
        np.ndarray: 经过预处理的图片
    """
    image = image.astype(np.float32)

    image = image.transpose(2, 0, 1)        # [H, W, C] -> [C, H, W]

    # openvino预处理会自动处理scale
    if not openvino_preprocess:
        image /= 255.0                      # 归一化

    input_array = np.expand_dims(image, 0)  # [C, H, W] -> [B, C, H, W]
    return input_array


def mulit_colors(num_classes: int):
    #---------------------------------------------------#
    #   https://github.com/bubbliiiing/yolov8-pytorch/blob/master/yolo.py#L88
    #   画框设置不同的颜色
    #---------------------------------------------------#
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    return colors


def check_onnx(onnx_path):
    """检查onnx模型是否损坏

    Args:
        onnx_path (str): onnx模型路径
    """
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


def nms(detections: np.ndarray, confidence_threshold: float=0.25, score_threshold: float=0.2, nms_threshold: float=0.45) -> np.ndarray:
    """后处理

    Args:
        detections (np.ndarray):                检测到的数据 [25200, 85]
        confidence_threshold (float, optional): prediction[4] 是否有物体得分阈值. Defaults to 0.25.
        score_threshold (float, optional):      nms分类得分阈值. Defaults to 0.2.
        nms_threshold (float, optional):        非极大值抑制iou阈值. Defaults to 0.45.

    Returns:
        (np.ndarray): np.float32
            [
                [class_index, confidences, xmin, ymin, xmax, ymax],
                ...
            ]
    """
    # boxes = []  # [[xmin, ymin, w, h]]
    # class_ids = []
    # confidences = []
    # 循环25200次,很慢,大约12ms
    # for prediction in detections:
    #     confidence = prediction[4].item()           # 是否有物体得分
    #     if confidence >= confidence_threshold:      # 是否有物体预支
    #         classes_scores = prediction[5:]         # 取出所有类别id
    #         class_id = np.argmax(classes_scores)    # 找到概率最大的id
    #         if (classes_scores[class_id] > .25):    # 最大概率必须大于 0.25
    #             confidences.append(confidence)      # 保存置信度(注意保存的是confidence，不是classes_scores[class_id]),类别id,box
    #             class_ids.append(class_id)
    #             # center_x, center_y, w, h
    #             x, y, w, h = prediction[0].item(), prediction[1].item(), prediction[2].item(), prediction[3].item()
    #             xmin = x - (w / 2)
    #             ymin = y - (h / 2)
    #             box = [xmin, ymin, w, h]
    #             boxes.append(box)

    # 加速优化写法
    # 通过置信度过滤一部分框
    detections = detections[detections[:, 4] > confidence_threshold]
    # 位置坐标
    loc = detections[:, :4]
    # 置信度
    confidences = detections[:, 4]
    # 分类
    cls = detections[:, 5:]
    # 最大分类index
    max_cls_index = cls.argmax(axis=-1)
    # 最大分类score
    max_cls_score = cls.max(axis=-1)
    # 置信度
    confidences = confidences[max_cls_score > .25]
    # 类别index
    class_index = max_cls_index[max_cls_score > .25]
    # 位置
    boxes = loc[max_cls_score > .25]
    # [center_x, center_y, w, h] -> [x_min, y_min, w, h]
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2

    # nms
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)

    # 根据nms返回的id获取对应的类别,置信度,box
    # detections = []
    # for i in indexes:
    #     j = i.item()
    #     boxes[j][2] += boxes[j][0] # w -> xmax
    #     boxes[j][3] += boxes[j][1] # h -> ymax
    #     detections.append({"class_index": class_index[j], "confidence": confidences[j], "box": boxes[j]})

    boxes[indexes, 2] += boxes[indexes, 0]  # w -> xmax
    boxes[indexes, 3] += boxes[indexes, 1]  # h -> ymax
    # [
    #   [class_index, confidences, xmin, ymin, xmax, ymax],
    #   ...
    # ]
    detections = np.concatenate((np.expand_dims(class_index[indexes], 1), np.expand_dims(confidences[indexes], 1), boxes[indexes]), axis=-1)

    return detections


def figure_boxes(detections: np.ndarray, delta_w: int,delta_h: int, size: list[int], image: np.ndarray, index2name: dict) -> np.ndarray:
    """将框画到原图

    Args:
        detections (np.ndarray): np.float32
                [
                    [class_index, confidences, xmin, ymin, xmax, ymax],
                    ...
                ]
        delta_w (int):      填充的宽
        delta_h (int):      填充的高
        size (list[int]):   推理 h w 640, 640
        image (np.ndarray): 原图
        index2name (dict):  index2name

    Returns:
        np.ndarray: 绘制的图
    """
    if len(detections) == 0:
        print("no detection")
        # 返回原图
        return image

    # 获取不同颜色
    colors = mulit_colors(len(index2name.keys()))

    # Print results and save Figure with detections
    for i, detection in enumerate(detections):
        classId = int(detection[0])
        confidence = detection[1]
        box = detection[2:]

        # 还原到原图尺寸并转化为int                    shape: (h, w)
        xmin = int(box[0] / ((size[1] - delta_w) / image.shape[1]))
        ymin = int(box[1] / ((size[0] - delta_h) / image.shape[0]))
        xmax = int(box[2] / ((size[1] - delta_w) / image.shape[1]))
        ymax = int(box[3] / ((size[0] - delta_h) / image.shape[0]))
        print( f"Bbox {i} Class: {classId}, Confidence: {'{:.2f}'.format(confidence)}, coords: [ xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax} ]" )

        # 绘制框
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[classId], 1)
        # 直接在原图上绘制文字背景，不透明
        # image = cv2.rectangle(image, (xmin, ymin - 20), (xmax, ymax)), colors[classId], cv2.FILLED)
        # 添加文字背景
        temp_image = np.zeros(image.shape).astype(np.uint8)
        temp_image = cv2.rectangle(temp_image, (xmin, ymin - 20), (xmax, ymin), colors[classId], cv2.FILLED)
        # 叠加原图和文字背景，文字背景是透明的
        image = cv2.addWeighted(image, 1.0, temp_image, 1.0, 1)
        # 添加文字
        image = cv2.putText(image, str(index2name[classId]) + " " + "{:.2f}".format(confidence),
                           (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    return image


def get_boxes(detections: np.ndarray, delta_w: int,delta_h: int, size: list[int], shape: np.ndarray) -> list:
    """将框还原到原图尺寸

    Args:
        detections (np.ndarray): np.float32
                [
                    [class_index, confidences, xmin, ymin, xmax, ymax],
                    ...
                ]
        delta_w (int):      填充的宽
        delta_h (int):      填充的高
        size (list[int]):   推理 h w 640, 640
        shape (np.ndarray): (h, w)

    Returns:
        res (list):  [{"class_index": class_index, "confidence": confidence, "box": [xmin, ymin, xmax, ymax]}， {}] box为int类型
    """
    if len(detections) == 0:
        print("no detection")
        # 返回原图
        return []

    res = []
    for detection in detections:
        box = [None] * 4
        # 还原到原图尺寸并转化为int                                          shape: (h, w)
        box[0] = int(detection[2] / ((size[1] - delta_w) / shape[1]))    # xmin
        box[1] = int(detection[3] / ((size[0] - delta_h) / shape[0]))    # ymin
        box[2] = int(detection[4] / ((size[1] - delta_w) / shape[1]))    # xmax
        box[3] = int(detection[5] / ((size[0] - delta_h) / shape[0]))    # ymax
        res.append({"class_index": int(detection[0]), "confidence": detection[1], "box": box})

    return res


if __name__ == "__main__":
    y = load_yaml("../weights/yolov5.yaml")
    print(y["size"])   # [640, 640]
    print(y["stride"]) # 32
    print(y["names"])  # {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
