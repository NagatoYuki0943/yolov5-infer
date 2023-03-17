from abc import ABC, abstractmethod
import numpy as np
import os
from .functions import *


class Inference(ABC):
    @abstractmethod
    def infer(self, image: np.ndarray) -> list[np.ndarray]:
        raise NotImplementedError


def single(inference: Inference, image_path: str, index2name: dict, confidence_threshold: float,
           score_threshold: float, nms_threshold: float, save_path: str, openvino_preprocess=False):
    """单张图片推理

    Args:
        inference (OrtInference):       推力器
        image_path (str):               图片路径
        index2name (dict):              index2name
        confidence_threshold (float):   只有得分大于置信度的预测框会被保留下来,越大越严格
        score_threshold (float):        框的得分置信度,越大越严格
        nms_threshold (float):          非极大抑制所用到的nms_iou大小,越小越严格
        save_path (str):                图片保存路径
        openvino_preprocess (bool, optional): 是否使用了openvino的图片预处理. Defaults to False.
    """

    # 1. 获取图片,缩放的图片,扩展的宽高
    image, image_reized, delta_w ,delta_h = get_image(image_path)
    input_array = transform(image_reized, openvino_preprocess)

    t1 = time.time()
    # 2. 推理
    boxes = inference.infer(input_array)
    t2 = time.time()

    print(boxes[0].shape)       # [1, 25200, 85]

    # 3. Postprocessing including NMS
    detections = boxes[0][0]    # [25200, 85]
    detections = nms(detections, confidence_threshold, score_threshold, nms_threshold)
    t3 = time.time()
    image = figure_boxes(detections, delta_w ,delta_h, image, index2name)
    t4 = time.time()
    print(f"infer time: {int((t2-t1) * 1000)} ms, nms time: {int((t3-t2) * 1000)} ms, figure time: {int((t4-t3) * 1000)} ms")

    # 4. 保存图片
    cv2.imwrite(save_path, image)


def multi(inference: Inference, image_dir: str, index2name: dict, confidence_threshold: float,
           score_threshold: float, nms_threshold: float, save_dir: str, openvino_preprocess=False):
    """单张图片推理

    Args:
        inference (OrtInference):       推力器
        image_dir (str):                图片文件夹路径
        index2name (dict):              index2name
        confidence_threshold (float):   只有得分大于置信度的预测框会被保留下来,越大越严格
        score_threshold (float):        框的得分置信度,越大越严格
        nms_threshold (float):          非极大抑制所用到的nms_iou大小,越小越严格
        save_dir (str):                 图片文件夹保存路径
        openvino_preprocess (bool, optional): 是否使用了openvino的图片预处理. Defaults to False.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1.获取文件夹中所有图片
    image_paths = os.listdir(image_dir)
    image_paths = [image for image in image_paths if image.lower().endswith(("jpg", "jepg", "bmp", "png"))]

    # 记录平均时间
    infer_times  = 0.0
    nms_times    = 0.0
    figure_times = 0.0

    # 2.遍历图片
    for image_file in image_paths:
        image_path = os.path.join(image_dir, image_file)

        # 3. 获取图片,缩放的图片,扩展的宽高
        image, image_reized, delta_w ,delta_h = get_image(image_path)
        input_array = transform(image_reized, openvino_preprocess)

        t1 = time.time()
        # 4. 推理
        boxes = inference.infer(input_array)
        t2 = time.time()

        print(boxes[0].shape)       # [1, 25200, 85]

        # 5. Postprocessing including NMS
        detections = boxes[0][0]    # [25200, 85]
        detections = nms(detections, confidence_threshold, score_threshold, nms_threshold)
        t3 = time.time()
        image = figure_boxes(detections, delta_w ,delta_h, image, index2name)
        t4 = time.time()

        # 6. 记录时间
        infer_time   = int((t2-t1) * 1000)
        nms_time     = int((t3-t2) * 1000)
        figure_time  = int((t4-t3) * 1000)
        infer_times  += infer_time
        nms_times    += nms_time
        figure_times += figure_times
        print(f"infer time: {infer_time} ms, nms time: {nms_time} ms, figure time: {figure_time} ms")

        # 7.保存图片
        cv2.imwrite(os.path.join(save_dir, image_file), image)

    print(f"avg infer time: {infer_times / len(image_paths)} ms, avg nms time: {nms_times / len(image_paths)} ms, avg figure time: {figure_times / len(image_paths)} ms")
