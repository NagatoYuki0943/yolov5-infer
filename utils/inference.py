from abc import ABC, abstractmethod
import numpy as np
import os
from collections import Counter
import logging, coloredlogs
from .functions import *


class Inference(ABC):
    def __init__(self,
                 yaml_path: str,
                 confidence_threshold: float = 0.25,
                 score_threshold:      float = 0.2,
                 nms_threshold:        float = 0.45,
                 openvino_preprocess         = False,
                 ) -> None:
        """父类推理器

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

        # logger
        self.logger: logging.Logger = logging.getLogger(name="Inference")

        # 保存log
        if not os.path.exists("./logs"):
            os.makedirs("./logs")
        logging.basicConfig(format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
                            filename="./logs/log.txt",
                            level=logging.DEBUG,
                            filemode="a")
        coloredlogs.install(level="DEBUG")
        # level: DEBUG, INFO, WARNING, ERROR, CRITICAL
        coloredlogs.install(level="DEBUG", logger=self.logger)


    @abstractmethod
    def infer(self, image: np.ndarray) -> list[np.ndarray]:
        """推理图片

        Args:
            image (np.ndarray): 图片

        Returns:
            list[np.ndarray]: 推理结果
        """
        raise NotImplementedError


    def nms(self, detections: np.ndarray) -> np.ndarray:
        """非极大值抑制,所有类别一起做的,没有分开做

        Args:
            detections (np.ndarray): 检测到的数据 [25200, 85]

        Returns:
            (np.ndarray): np.float32
                [
                    [class_index, confidences, xmin, ymin, xmax, ymax],
                    ...
                ]
        """
        # 加速优化写法
        # 通过置信度过滤一部分框
        detections     = detections[detections[:, 4] > self.confidence_threshold]
        # 位置坐标
        loc            = detections[:, :4]
        # 置信度
        confidences    = detections[:, 4]
        # 分类
        cls            = detections[:, 5:]
        # 最大分类index
        max_cls_index  = cls.argmax(axis=-1)
        # 最大分类score
        max_cls_score  = cls.max(axis=-1)

        # 位置
        boxes          = loc[max_cls_score > .25]
        # 置信度
        confidences    = confidences[max_cls_score > .25]
        # 类别index
        class_index    = max_cls_index[max_cls_score > .25]

        # [center_x, center_y, w, h] -> [x_min, y_min, w, h]
        boxes[:, 0:2] -= boxes[:, 2:4] / 2

        # nms
        nms_indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.score_threshold, self.nms_threshold)

        # nms过滤
        boxes = boxes[nms_indexes]
        # [x_min, y_min, w, h] -> [x_min, y_min, x_max, y_max]
        boxes[:, 2:4] += boxes[:, 0:2]

        # 防止框超出图片边界, 前面判断为True/False,后面选择更改的列,不选择更改的列会将整行都改为0
        boxes[boxes[:, 0] < 0.0, 0] = 0.0
        boxes[boxes[:, 1] < 0.0, 1] = 0.0
        boxes[boxes[:, 2] > self.config["size"][1], 2] = self.config["size"][1]
        boxes[boxes[:, 3] > self.config["size"][0], 3] = self.config["size"][0]

        # [
        #   [class_index, confidences, xmin, ymin, xmax, ymax],
        #   ...
        # ]
        detections = np.concatenate((np.expand_dims(class_index[nms_indexes], 1), np.expand_dims(confidences[nms_indexes], 1), boxes), axis=-1)
        return detections


    def figure_boxes(self, detections: np.ndarray, delta_w: int, delta_h: int, image: np.ndarray, ignore_overlap_box: bool = False) -> np.ndarray:
        """将框画到原图

        Args:
            detections (np.ndarray): np.float32
                    [
                        [class_index, confidences, xmin, ymin, xmax, ymax],
                        ...
                    ]
            delta_w (int):      填充的宽
            delta_h (int):      填充的高
            image (np.ndarray): 原图
            ignore_overlap_box (bool, optional): 是否忽略重叠的小框,不同于nms. Defaults to False.

        Returns:
            np.ndarray: 绘制的图
        """
        if len(detections) == 0:
            self.logger.warning("no detection")
            # 返回原图
            return image

        # 忽略重叠的小框,不同于nms
        if ignore_overlap_box:
            detections = ignore_overlap_boxes(detections)

        # 获取不同颜色
        colors = mulit_colors(len(self.config["names"].keys()))

        # Print results and save Figure with detections
        for i, detection in enumerate(detections):
            classId = int(detection[0])
            confidence = detection[1]
            box = detection[2:]

            # 还原到原图尺寸并转化为int                    shape: (h, w)
            xmin = int(box[0] / ((self.config["size"][1] - delta_w) / image.shape[1]))
            ymin = int(box[1] / ((self.config["size"][0] - delta_h) / image.shape[0]))
            xmax = int(box[2] / ((self.config["size"][1] - delta_w) / image.shape[1]))
            ymax = int(box[3] / ((self.config["size"][0] - delta_h) / image.shape[0]))
            self.logger.info(f"Bbox {i} Class: {classId}, Confidence: {'{:.2f}'.format(confidence)}, coords: [ xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax} ]")

            # 绘制框
            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[classId], 2)
            # 直接在原图上绘制文字背景，不透明
            # image = cv2.rectangle(image, (xmin, ymin - 20), (xmax, ymax)), colors[classId], cv2.FILLED)

            # 文字
            label = str(self.config["names"][classId]) + " " + "{:.2f}".format(confidence)
            w, h = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]  # text width, height

            # 添加文字背景
            temp_image = np.zeros(image.shape).astype(np.uint8)
            temp_image = cv2.rectangle(temp_image, (xmin, ymin - 20 if ymin > 20 else ymin + h + 10), (xmax, ymin), colors[classId], cv2.FILLED)
            # 叠加原图和文字背景，文字背景是透明的
            image = cv2.addWeighted(image, 1.0, temp_image, 1.0, 1)

            # 添加文字
            image = cv2.putText(img         = image,
                                text        = label,
                                org         = (xmin, ymin - 5 if ymin > 20 else ymin + h + 5),
                                fontFace    = 0,
                                fontScale   = 0.5,
                                color       = (0, 0, 0),
                                thickness   = 1,
                                lineType    = cv2.LINE_AA,
                                )

        return image


    def get_boxes(self, detections: np.ndarray, delta_w: int, delta_h: int, shape: np.ndarray, ignore_overlap_box: bool = False) -> dict:
        """返回还原到原图的框

        Args:
            detections (np.ndarray): np.float32
                    [
                        [class_index, confidences, xmin, ymin, xmax, ymax],
                        ...
                    ]
            delta_w (int):      填充的宽
            delta_h (int):      填充的高
            shape (np.ndarray): (h, w, c)
            ignore_overlap_box (bool, optional): 是否忽略重叠的小框,不同于nms. Defaults to False.

        Returns:
            detect (dict):  {
                            "detect":     [{"class_index": class_index, "class": "class_name", "confidence": confidence, "box": [xmin, ymin, xmax, ymax]}...],    box为int类型
                            "num":        {"Person": 4, "Bus": 1},
                            "image_size": [height, width, Channel]
                            }
        """
        if len(detections) == 0:
            self.logger.warning("no detection")
            return {"detect": [], "num": {}, "image_size": shape}

        # 忽略重叠的小框,不同于nms
        if ignore_overlap_box:
            detections = ignore_overlap_boxes(detections)

        detect = {} # 结果返回一个dict
        count = []  # 类别计数
        res = []
        for detection in detections:
            count.append(int(detection[0]))   # 计数
            box = [None] * 4
            # 还原到原图尺寸并转化为int                                          shape: (h, w)
            box[0] = int(detection[2] / ((self.config["size"][1] - delta_w) / shape[1]))    # xmin
            box[1] = int(detection[3] / ((self.config["size"][0] - delta_h) / shape[0]))    # ymin
            box[2] = int(detection[4] / ((self.config["size"][1] - delta_w) / shape[1]))    # xmax
            box[3] = int(detection[5] / ((self.config["size"][0] - delta_h) / shape[0]))    # ymax
            res.append({"class_index": int(detection[0]), "class": self.config["names"][int(detection[0])], "confidence": detection[1], "box": box})
        detect["detect"] = res
        # 类别计数
        detect["num"] = dict(Counter(count))
        # 图片形状
        detect["image_size"] = shape # 添加 (h, w, c)
        return detect


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
        detections = self.nms(detections)
        t4 = time.time()
        image = self.figure_boxes(detections, delta_w, delta_h, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        t5 = time.time()
        self.logger.info(f"transform time: {int((t2-t1) * 1000)} ms, infer time: {int((t3-t2) * 1000)} ms, nms time: {int((t4-t3) * 1000)} ms, figure time: {int((t5-t4) * 1000)} ms")

        # 4. 返回图片
        return image


    def single_get_boxes(self, image_rgb: np.ndarray) -> dict:
        """单张图片推理
        Args:
            image_path (str):   图片路径

        Returns:
            detect (dict):  {
                            "detect":     [{"class_index": class_index, "class": "class_name", "confidence": confidence, "box": [xmin, ymin, xmax, ymax]}...],    box为int类型
                            "num":        {"Person": 4, "Bus": 1},
                            "image_size": [height, width, Channel]
                            }
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
        detections = self.nms(detections)
        t4 = time.time()
        detect = self.get_boxes(detections, delta_w, delta_h, image_rgb.shape) # shape: (h, w, c)
        t5 = time.time()

        self.logger.info(f"transform time: {int((t2-t1) * 1000)} ms, infer time: {int((t3-t2) * 1000)} ms, nms time: {int((t4-t3) * 1000)} ms, get boxes time: {int((t5-t4) * 1000)} ms")

        # 4. 返回detect
        return detect


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
            detections = self.nms(detections)
            t4 = time.time()
            image = self.figure_boxes(detections, delta_w, delta_h, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
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
            self.logger.info(f"transform time: {trans_time} ms, infer time: {infer_time} ms, nms time: {nms_time} ms, figure time: {figure_time} ms")

            # 7.保存图片
            cv2.imwrite(os.path.join(save_dir, image_file), image)

        self.logger.info(f"avg transform time: {trans_times / len(image_paths)} ms, avg infer time: {infer_times / len(image_paths)} ms, avg nms time: {nms_times / len(image_paths)} ms, avg figure time: {figure_times / len(image_paths)} ms")
