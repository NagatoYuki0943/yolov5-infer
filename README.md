# 运行yolov5导出的onnx,engine,openvino等

下载pt,onnx地址 [Releases · ultralytics/yolov5 (github.com)](https://github.com/ultralytics/yolov5/releases)

# 参考

https://github.com/dacquaviva/yolov5-openvino-cpp-python

# 文件

1. 需要权重，如onnx，tensorrt，openvino等

2. 需要配置文件，如下格式

   必须要有 `size` 和 `names`

```yaml
# infer size
size:
  - 640 # height
  - 640 # width

# down sample stride
stride: 32

# classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
```



# Onnxruntime推理例子

```python
from onnxruntime_infer import OrtInference
from utils import get_image
import cv2


ONNX_PATH  = "./weights/yolov5s.onnx"
config = {
    'yaml_path':            "./weights/yolov5.yaml",
    'confidence_threshold': 0.25,   # 只有得分大于置信度的预测框会被保留下来,越大越严格
    'score_threshold':      0.2,    # nms分类得分阈值,越大越严格
    'nms_threshold':        0.45,   # 非极大抑制所用到的nms_iou大小,越小越严格
}

# 实例化推理器
inference = OrtInference(model_path=ONNX_PATH, mode="cpu", **config)


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
```

# OpenVINO推理例子

```python
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
```

# TensorRT推理例子

```python
from tensorrt_infer import TensorRTInfer
from utils import get_image
import cv2


ENGINE_PATH  = "./weights/yolov5s.engine"
config = {
    'yaml_path':            "./weights/yolov5.yaml",
    'confidence_threshold': 0.25,   # 只有得分大于置信度的预测框会被保留下来,越大越严格
    'score_threshold':      0.2,    # nms分类得分阈值,越大越严格
    'nms_threshold':        0.45,   # 非极大抑制所用到的nms_iou大小,越小越严格
}

# 实例化推理器
inference = TensorRTInfer(model_path=ENGINE_PATH, **config)

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
```

