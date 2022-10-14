"""onnx图片需要调整BGR2RGB, 并调整通道为[B, C, H, W], 且需要归一化
"""

from pathlib import Path

import onnx
import onnxruntime as ort

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
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR2RGB

    img_reized, delta_w ,delta_h = resize_and_pad(img_rgb, (640, 640))

    img_reized = img_reized.astype(np.float32)
    img_reized /= 255.0                             # 归一化

    img_reized = img_reized.transpose(2, 0, 1)      # [H, W, C] -> [C, H, W]
    input_tensor = np.expand_dims(img_reized, 0)    # [C, H, W] -> [B, C, H, W]

    return img, input_tensor, delta_w ,delta_h


def get_onnx_model(onnx_path, cuda=False):
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
    ONNX_PATH  = "../weights/yolov5s.onnx"
    IMAGE_PATH = "../images/bus.jpg"
    YAML_PATH  = "../weights/yolov5s.yaml"

    # 获取图片,扩展的宽高
    img, input_tensor, delta_w ,delta_h = get_image(IMAGE_PATH)

    # 获取模型
    model = get_onnx_model(ONNX_PATH, False)

    # 获取label
    index2label = get_index2label(YAML_PATH)

    start = time.time()
    detections = model.run(None, {model.get_inputs()[0].name: input_tensor})
    # print(detections[0].shape)                        # [1, 25200, 85]
    detections = np.squeeze(detections[0])              # [25200, 85]

    # Step 8. Postprocessing including NMS
    img = post(detections, delta_w ,delta_h, img, CONFIDENCE_THRESHOLD, SCORE_THRESHOLD, NMS_THRESHOLD, index2label)
    end = time.time()
    print((end - start) * 1000)

    cv2.imwrite("./onnx_det.png", img)


if __name__ == "__main__":
    inference()
