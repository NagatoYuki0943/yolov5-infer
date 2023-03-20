from openvino_infer import OVInference
from utils import get_image
import cv2


ONNX_PATH  = "./weights/yolov5s_openvino_model/yolov5s.xml"
config = {
    'yaml_path':            "./weights/yolov5.yaml",
    'confidence_threshold': 0.25,   # 只有得分大于置信度的预测框会被保留下来,越大越严格
    'score_threshold':      0.2,    # nms分类得分阈值,越大越严格
    'nms_threshold':        0.45,   # 非极大抑制所用到的nms_iou大小,越小越严格
    'openvino_preprocess':  True,   # 是否使用openvino图片预处理
}

# 实例化推理器
inference = OVInference(model_path=ONNX_PATH, mode="cpu", **config)

# 读取图片
IMAGE_PATH = "./images/bus.jpg"
image_rgb = get_image(IMAGE_PATH)

# 单张图片推理
res = inference.single(image_rgb)
cv2.imshow("res", res)
cv2.waitKey(0)
print(inference.single_get_boxes(image_rgb))

# 多张图片推理
IMAGE_DIR = "../datasets/coco128/images/train2017"
SAVE_DIR  = "../datasets/coco128/images/train2017_res"
# inference.multi(IMAGE_DIR, SAVE_DIR)
