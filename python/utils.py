from stringprep import in_table_d2
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


def post(detections, delta_w ,delta_h, img, confidence_threshold, score_threshold, nms_threshold, index2label):
    """后处理

    Args:
        detections (np.Array): 检测到的数据 [25200, 85]
        delta_w (int):  填充的宽
        delta_h (int):  填充的高
        img (np.Array): 原图
        confidence_threshold (float): prediction[4] 是否有物体得分阈值
        score_threshold (float):      分类得分阈值
        nms_threshold (float):        非极大值抑制iou阈值
        index2label (dict):           id2label
    Returns:
        np.Array: 绘制好的图片
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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)

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
        img = cv2.putText(img, str(index2label[classId]) + " " + "{:.2f}".format(confidence),
                          (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    return img


def get_index2label(yaml_path):
    """通过id找到名称

    Args:
        yaml_path (str): yaml文件路径

    Returns:
        dict: id2name dict
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        y = yaml.load(f, Loader=yaml.FullLoader)

    return y['names']


if __name__ == "__main__":
    index2label = get_index2label("./weights/yolov5.yaml")
    print(index2label[0])   # person
