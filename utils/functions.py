import sys
sys.path.append("../")

import yaml
import cv2
import onnx
import time
import numpy as np
import colorsys
from pathlib import Path
import logging


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
    # 填充bottom和right的长度
    delta_w = new_shape[1] - new_size[1]
    delta_h = new_shape[0] - new_size[0]
    # 使用灰色填充到640*640的形状
    color = [100, 100, 100]
    #                                 src, top, bottom, left, right
    image_reized = cv2.copyMakeBorder(image, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT, value=color)

    return image_reized, delta_w ,delta_h


def transform(image: np.ndarray, openvino_preprocess = False) -> np.ndarray:
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
    #              hue saturation value
    hsv_tuples = [(x / num_classes, 0.7, 1.) for x in range(num_classes)]
    # colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = [colorsys.hsv_to_rgb(*x) for x in hsv_tuples]
    # colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    colors = [(int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)) for x in colors]
    return colors


def check_onnx(onnx_path, logger: logging.Logger):
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
        logger.error("Model incorrect")
    else:
        logger.info("Model correct")


def np_softmax(array: np.ndarray, axis=-1) -> np.ndarray:
    array -= np.max(array)
    array = np.exp(array)
    print(array)
    return array / np.sum(array, axis=axis)


def find_inner_box_isin_outer_box(box1: list, box2: list, ratio: float = 0.75) -> bool:
    """determine whether a box is in another box

    Args:
        box1 (list): 假设外部盒子 [x_min, y_min, x_max, y_max]
        box2 (list): 假设内部盒子 [x_min, y_min, x_max, y_max]
        ratio (float): inner_box相当于box2的面积的阈值,大于阈值就忽略. Defaults to 0.75.

    Returns:
        bool: 外部盒子是否包含内部盒子
    """
    # 内部盒子面积
    inner_box_x1 = max(box1[0], box2[0])
    inner_box_y1 = max(box1[1], box2[1])
    inner_box_x2 = min(box1[2], box2[2])
    inner_box_y2 = min(box1[3], box2[3])
    # max 用来判断是否重叠
    inner_box_area = max(inner_box_x2 - inner_box_x1, 0) * max(inner_box_y2 - inner_box_y1, 0)

    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    if inner_box_area / box2_area > ratio:
        return True
    else:
        return False


def ignore_overlap_boxes(detections: np.ndarray) -> np.ndarray:
    """忽略一些框,根据同一个类别的框是否包含另一个框

    Args:
        detections (np.ndarray): np.float32
                [
                    [class_index, confidences, xmin, ymin, xmax, ymax],
                    ...
                ]

    Returns:
                [
                    [class_index, confidences, xmin, ymin, xmax, ymax],
                    ...
                ]
    """
    new_detections = []

    # 获取每个类别
    classes = np.unique(detections[:, 0])
    # 遍历单一类别
    for cls in classes:
        dets_sig_cls = detections[detections[:, 0] == cls]
        # 如果一个类别只有1个框,就直接保存
        if len(dets_sig_cls) == 1:
            new_detections.append(dets_sig_cls)
            continue
        # 求面积,根据面积排序,不是最好的办法
        h = dets_sig_cls[:, 5] - dets_sig_cls[:, 3]
        w = dets_sig_cls[:, 4] - dets_sig_cls[:, 2]
        area = np.array(h * w)
        index = area.argsort()  # 得到面积排序index
        index = index[::-1]     # 转换为降序

        # max代表大的框,min代表小的框
        keeps = []
        for i, max in enumerate(index[:-1]):
            # 默认都不包含
            keep = [False] * len(dets_sig_cls)
            for min in index[i+1:]:
                isin = find_inner_box_isin_outer_box(dets_sig_cls[max, 2:], dets_sig_cls[min, 2:])
                keep[min] = isin # 包含
            keeps.append(keep)
        # 取反,原本False为不包含,True为包含,取反后False为不保留,True为保留
        keeps = ~np.array(keeps)
        # print(keeps) # 每一行代表被判断的框相对于判断框是否要保留
        # [[True, True, True, True, False, True,  True,  True, True,  True,  True,  False],
        #  [True, True, True, True, True,  True,  True,  True, True,  True,  True,  True],
        #  [True, True, True, True, True,  True,  False, True, True,  True,  False, True],
        #  [True, True, True, True, True,  False, True,  True, False, False, True,  True],
        #  [True, True, True, True, True,  True,  True,  True, True,  True,  True,  True],
        #  [True, True, True, True, True,  True,  True,  True, True,  True,  False, True],
        #  [True, True, True, True, True,  True,  True,  True, True,  True,  True,  True],
        #  [True, True, True, True, True,  True,  True,  True, True,  True,  True,  True],
        #  [True, True, True, True, True,  True,  True,  True, True,  True,  True,  True],
        #  [True, True, True, True, True,  True,  True,  True, True,  True,  True,  True],
        #  [True, True, True, True, True,  True,  True,  True, True,  True,  True,  True]]

        # 最终保留的index,True/False
        # keeps.T: 转置之后每行代表是否要保留这个框
        final_keep = np.all(keeps.T, axis=-1)
        new_detections.append(dets_sig_cls[final_keep])

    # new_detections：[np.ndarray, np.ndarray...]
    return np.concatenate(new_detections, axis=0)


if __name__ == "__main__":
    # y = load_yaml("../weights/yolov5.yaml")
    # print(y["size"])   # [640, 640]
    # print(y["stride"]) # 32
    # print(y["names"])  # {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

    detections = np.array([[10, 0.8, 200.04971, 196.26697, 489.98325, 424.07892],
                           [10, 0.7, 141.04881, 311.3442 , 228.94856, 408.5379 ],
                           [10, 0.6, 0.       , 303.4387 , 175.52124, 424.90558],
                           [10, 0.5, 176.42613, 0.       , 460.68604, 227.06232],
                           [10, 0.3, 384.6766 , 283.063  , 419.97977, 335.35898],
                           [10, 0.8, 97.71875 , 346.97867, 103.96518, 353.037  ],
                           [10, 0.7, 575.25476, 195.62448, 628.17926, 291.2721 ],
                           [10, 0.6, 450.49182, 1.8310547, 640.     , 292.99066],
                           [10, 0.7, 73.79396 , 368.1626 , 79.10231 , 372.40448],
                           [10, 0.9, 84.013214, 332.34296, 89.18914 , 337.10605],
                           [10, 0.8, 596.2429 , 248.21837, 601.9428 , 253.99461],
                           [10, 0.1, 372.0439 , 363.4396 , 378.0838 , 368.31393]])
    print(len(detections))      # 12
    new_detections = ignore_overlap_boxes(detections)
    print(len(new_detections))  # 5
