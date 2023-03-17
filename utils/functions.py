import sys
sys.path.append("../")

import yaml
import cv2
import onnx
import numpy as np


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


def nms(detections: np.ndarray, confidence_threshold: float, score_threshold: float, nms_threshold: float) -> list:
    """后处理

    Args:
        detections (np.ndarray): 检测到的数据 [25200, 85]
        confidence_threshold (float): prediction[4] 是否有物体得分阈值
        score_threshold (float):      分类得分阈值
        nms_threshold (float):        非极大值抑制iou阈值

    Returns:
        list: 经过mns处理的框
    """
    boxes = []
    class_ids = []
    confidences = []
    for prediction in detections:
        confidence = prediction[4].item()
        if confidence >= confidence_threshold:
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

    # nms
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)

    detections = []
    for i in indexes:
        j = i.item()
        detections.append({"class_index": class_ids[j], "confidence": confidences[j], "box": boxes[j]})

    return detections


def figure_boxes(detections: list, delta_w: int,delta_h: int, img: np.ndarray, index2label: dict) -> np.ndarray:
    """将框画到原图

    Args:
        detections (list):  经过mns处理的框
        delta_w (int):      填充的宽
        delta_h (int):      填充的高
        img (np.ndarray):   原图
        index2label (dict): id2label

    Returns:
        np.ndarray: 绘制的图
    """
    if len(detections) == 0:
        print("no detection")
        return img

# Print results and save Figure with detections
    for i, detection in enumerate(detections):
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
        img = cv2.putText(img, str(index2label[classId]) + " " + "{:.2f}".format(confidence),
                          (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    return img


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


if __name__ == "__main__":
    y = load_yaml("../weights/yolov5.yaml")
    print(y["size"])   # [640, 640]
    print(y["stride"]) # 32
    print(y["names"])  # {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
