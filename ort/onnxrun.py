"""onnx图片需要调整BGR2RGB, 并调整通道为[B, C, H, W], 且需要归一化
"""

import sys
sys.path.append("../")

from pathlib import Path
import onnxruntime as ort
import numpy as np
import cv2
import time
from utils import get_image, check_onnx, nms, figure_boxes, load_yaml


CONFIDENCE_THRESHOLD = 0.25 # 只有得分大于置信度的预测框会被保留下来,越大越严格
SCORE_THRESHOLD      = 0.2  # 框的得分置信度,越大越严格
NMS_THRESHOLD        = 0.45 # 非极大抑制所用到的nms_iou大小,越小越严格


def get_onnx_model(onnx_path: str, cuda=False):
    """获取模型

    Args:
        onnx_path (str): 模型路径
        cuda (bool, optional): 是否使用cuda. Defaults to False.

    Returns:
        InferenceSession: 推理模型
    """
    check_onnx(onnx_path)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    # 新版本onnxruntime-gpu这样指定cpu和gpu
    if cuda:
        model = ort.InferenceSession(onnx_path, sess_options=so, providers=['CUDAExecutionProvider'])
    else:
        model = ort.InferenceSession(onnx_path, sess_options=so, providers=['CPUExecutionProvider'])

    #--------------------------------#
    #   查看model中的内容
    #   get_inputs()返回对象，[0]返回名字
    #--------------------------------#
    # print("model outputs: \n", model.get_inputs())    # 列表 [<onnxruntime.capi.onnxruntime_pybind11_state.NodeArg object at 0x000001B16D601730>]
    # print(model.get_inputs()[0])                      # NodeArg(name='images', type='tensor(float)', shape=[1, 3, 640, 640])
    # print(model.get_inputs()[0].name)                 # images
    # print(model.get_inputs()[0].type)                 # tensor(float)
    # print(model.get_inputs()[0].shape, "\n")          # [1, 3, 640, 640]

    # print("model outputs: \n", model.get_outputs())   # 列表 [<onnxruntime.capi.onnxruntime_pybind11_state.NodeArg object at 0x000001B16D6016B0>, <onnxruntime.capi.onnxruntime_pybind11_state.NodeArg object at 0x000001B16D6017B0>, <onnxruntime.capi.onnxruntime_pybind11_state.NodeArg object at 0x000001B16D6017F0>, <onnxruntime.capi.onnxruntime_pybind11_state.NodeArg object at 0x000001B16D601830>]
    # print(model.get_outputs()[0])                     # NodeArg(name='output', type='tensor(float)', shape=[1, 25200, 85])
    # print(model.get_outputs()[0].name)                # output
    # print(model.get_outputs()[0].type)                # tensor(float)
    # print(model.get_outputs()[0].shape, "\n")         # [1, 25200, 85]

    return model


#--------------------------------#
#   推理
#--------------------------------#
def inference():
    YAML_PATH  = "../weights/yolov5.yaml"
    ONNX_PATH  = "../weights/yolov5s.onnx"
    IMAGE_PATH = "../images/bus.jpg"

    # 1. 获取图片,缩放的图片,扩展的宽高
    img, input_tensor, delta_w ,delta_h = get_image(IMAGE_PATH)

    # 2. 获取模型
    model = get_onnx_model(ONNX_PATH, False)

    # 3. 获取label
    y = load_yaml(YAML_PATH)
    index2name = y["names"]

    start = time.time()
    # 4. infer 返回一个列表,每一个数据是一个3维numpy数组
    boxes = model.run(None, {model.get_inputs()[0].name: input_tensor})
    print(boxes[0].shape)          # [1, 25200, 85]
    detections = boxes[0][0]        # [25200, 85]

    # 5. Postprocessing including NMS
    detections = nms(detections, CONFIDENCE_THRESHOLD, SCORE_THRESHOLD, NMS_THRESHOLD)
    img = figure_boxes(detections, delta_w ,delta_h, img, index2name)
    end = time.time()
    print(f'time: {int((end - start) * 1000)} ms')

    cv2.imwrite("./onnx_det.png", img)


if __name__ == "__main__":
    inference()
