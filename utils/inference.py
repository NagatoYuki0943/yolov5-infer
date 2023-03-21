from abc import ABC, abstractmethod
import numpy as np
import os
from .functions import *


class Inference(ABC):
    def __init__(self,
                 yaml_path: str,
                 confidence_threshold: float,
                 score_threshold: float,
                 nms_threshold: float,
                 openvino_preprocess=False,
                 ) -> None:
        """推力器

        Args:
            yaml_path (str):                配置文件路径
            confidence_threshold (float):   只有得分大于置信度的预测框会被保留下来,越大越严格
            score_threshold (float):        nms分类得分阈值,越大越严格
            nms_threshold (float):          非极大抑制所用到的nms_iou大小,越小越严格
            openvino_preprocess (bool, optional): openvino图片预处理，只有openvino模型可用. Defaults to False.
        """
        self.config               = load_yaml(yaml_path)
        self.confidence_threshold = confidence_threshold
        self.score_threshold      = score_threshold
        self.nms_threshold        = nms_threshold
        self.openvino_preprocess  = openvino_preprocess

    @abstractmethod
    def infer(self, image: np.ndarray) -> list[np.ndarray]:
        """推理图片

        Args:
            image (np.ndarray): 图片

        Returns:
            list[np.ndarray]: 推理结果
        """
        raise NotImplementedError

    def single(self, image_rgb: np.ndarray) -> np.ndarray:
        """单张图片推理
        Args:
            image_rgb (np.ndarray):   rgb图片

        Returns:
            np.ndarray: 绘制好的图片
        """

        # 1. 缩放图片,扩展的宽高
        t1 = time.time()
        image_reized, delta_w ,delta_h = resize_and_pad(image_rgb, self.config["size"])
        input_array = transform(image_reized, self.openvino_preprocess)

        # 2. 推理
        t2 = time.time()
        boxes = self.infer(input_array)
        # print(boxes[0].shape)       # [1, 25200, 85]

        # 3. Postprocessing including NMS
        t3 = time.time()
        detections = boxes[0][0]    # [25200, 85]
        detections = nms(detections, self.confidence_threshold, self.score_threshold, self.nms_threshold)
        t4 = time.time()
        image = figure_boxes(detections, delta_w ,delta_h, self.config["size"], cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), self.config["names"])
        t5 = time.time()
        print(f"transform time: {int((t2-t1) * 1000)} ms, infer time: {int((t3-t2) * 1000)} ms, nms time: {int((t4-t3) * 1000)} ms, figure time: {int((t5-t4) * 1000)} ms")

        # 4. 返回图片
        return image

    def single_get_boxes(self, image_rgb: np.ndarray) -> np.ndarray:
        """单张图片推理
        Args:
            image_path (str):   图片路径

        Returns:
            np.ndarray: 绘制好的图片
        """

        # 1. 缩放的图片,扩展的宽高
        t1 = time.time()
        image_reized, delta_w ,delta_h = resize_and_pad(image_rgb, self.config["size"])
        input_array = transform(image_reized, self.openvino_preprocess)

        # 2. 推理
        t2 = time.time()
        boxes = self.infer(input_array)
        # print(boxes[0].shape)       # [1, 25200, 85]

        # 3. Postprocessing including NMS
        t3 = time.time()
        detections = boxes[0][0]    # [25200, 85]
        detections = nms(detections, self.confidence_threshold, self.score_threshold, self.nms_threshold)
        t4 = time.time()
        boxes = get_boxes(detections, delta_w ,delta_h, self.config["size"], image_rgb.shape) # shape: (h, w)
        t5 = time.time()

        print(f"transform time: {int((t2-t1) * 1000)} ms, infer time: {int((t3-t2) * 1000)} ms, nms time: {int((t4-t3) * 1000)} ms, get boxes time: {int((t5-t4) * 1000)} ms")

        # 4. 返回boxes
        return boxes

    def multi(self, image_dir: str, save_dir: str):
        """单张图片推理

        Args:
            image_dir (str):    图片文件夹路径
            save_dir (str):     图片文件夹保存路径
        """
        if not os.path.exists(save_dir):
            print(f"The save path {save_dir} does not exist, it has been created")
            os.makedirs(save_dir)

        # 1.获取文件夹中所有图片
        image_paths = os.listdir(image_dir)
        image_paths = [image for image in image_paths if image.lower().endswith(("jpg", "jepg", "bmp", "png"))]

        # 记录平均时间
        trans_times  = 0.0
        infer_times  = 0.0
        nms_times    = 0.0
        figure_times = 0.0

        # 2.遍历图片
        for image_file in image_paths:
            image_path = os.path.join(image_dir, image_file)

            # 3. 获取图片,缩放的图片,扩展的宽高
            t1 = time.time()
            image_rgb = get_image(image_path)
            image_reized, delta_w ,delta_h = resize_and_pad(image_rgb, self.config["size"])
            input_array = transform(image_reized, self.openvino_preprocess)

            # 4. 推理
            t2 = time.time()
            boxes = self.infer(input_array)
            # print(boxes[0].shape)       # [1, 25200, 85]

            # 5. Postprocessing including NMS
            t3 = time.time()
            detections = boxes[0][0]    # [25200, 85]
            detections = nms(detections, self.confidence_threshold, self.score_threshold, self.nms_threshold)
            t4 = time.time()
            image = figure_boxes(detections, delta_w ,delta_h, self.config["size"], cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), self.config["names"])
            t5 = time.time()

            # 6. 记录时间
            trans_time   = int((t2-t1) * 1000)
            infer_time   = int((t3-t2) * 1000)
            nms_time     = int((t4-t3) * 1000)
            figure_time  = int((t5-t4) * 1000)
            trans_times  += trans_time
            infer_times  += infer_time
            nms_times    += nms_time
            figure_times += figure_times
            print(f"transform time: {trans_time} ms, infer time: {infer_time} ms, nms time: {nms_time} ms, figure time: {figure_time} ms")

            # 7.保存图片
            cv2.imwrite(os.path.join(save_dir, image_file), image)

        print(f"avg transform time: {trans_times / len(image_paths)} ms, avg infer time: {infer_times / len(image_paths)} ms, avg nms time: {nms_times / len(image_paths)} ms, avg figure time: {figure_times / len(image_paths)} ms")
