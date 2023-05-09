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
imgsz:
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

> `onnxruntime-gpu` 使用显卡要使用 `cuda` 和 `cudnn

```python
from onnxruntime_infer import OrtInference
from utils import get_image
import cv2


config = {
    "model_path":           r"./weights/yolov5s.onnx",
    "mode":                 r"cuda",
    "yaml_path":            r"./weights/yolov5.yaml",
    "confidence_threshold": 0.25,   # 只有得分大于置信度的预测框会被保留下来,越大越严格
    "score_threshold":      0.2,    # opencv nms分类得分阈值,越大越严格
    "nms_threshold":        0.45,   # 非极大抑制所用到的nms_iou大小,越小越严格
}

# 实例化推理器
inference = OrtInference(**config)

# 读取图片
IMAGE_PATH = r"./images/bus.jpg"
image_rgb = get_image(IMAGE_PATH)

# 单张图片推理
result, image_bgr_detect = inference.single(image_rgb, only_get_boxes=False)
print(result)
cv2.imshow("res", image_bgr_detect)
cv2.waitKey(0)

# 多张图片推理
IMAGE_DIR = r"../datasets/coco128/images/train2017"
SAVE_DIR  = r"../datasets/coco128/images/train2017_res"
# inference.multi(IMAGE_DIR, SAVE_DIR, save_xml=True) # save_xml 保存xml文件
```

# OpenVINO推理例子

> 安装openvino方法请看openvino文件夹的`readme.md`

```python
from openvino_infer import OVInference
from utils import get_image
import cv2


config = {
    "model_path":           r"./weights/yolov5s_openvino_model/yolov5s.xml",
    "mode":                 r"cpu",
    "yaml_path":            r"./weights/yolov5.yaml",
    "confidence_threshold": 0.25,   # 只有得分大于置信度的预测框会被保留下来,越大越严格
    "score_threshold":      0.2,    # opencv nms分类得分阈值,越大越严格
    "nms_threshold":        0.45,   # 非极大抑制所用到的nms_iou大小,越小越严格
    "openvino_preprocess":  True,   # 是否使用openvino图片预处理
}

# 实例化推理器
inference = OVInference(**config)

# 读取图片
IMAGE_PATH = r"./images/bus.jpg"
image_rgb = get_image(IMAGE_PATH)

# 单张图片推理
result, image_bgr_detect = inference.single(image_rgb, only_get_boxes=False)
print(result)
cv2.imshow("res", image_bgr_detect)
cv2.waitKey(0)

# 多张图片推理
IMAGE_DIR = r"../datasets/coco128/images/train2017"
SAVE_DIR  = r"../datasets/coco128/images/train2017_res"
# inference.multi(IMAGE_DIR, SAVE_DIR, save_xml=True) # save_xml 保存xml文件
```

# TensorRT推理例子

> 安装tensorrt方法请看tensorrt文件夹的`readme.md`

```python
from tensorrt_infer import TensorRTInfer
from utils import get_image
import cv2


config = {
    "model_path":           r"./weights/yolov5s.engine",
    "yaml_path":            r"./weights/yolov5.yaml",
    "confidence_threshold": 0.25,   # 只有得分大于置信度的预测框会被保留下来,越大越严格
    "score_threshold":      0.2,    # opencv nms分类得分阈值,越大越严格
    "nms_threshold":        0.45,   # 非极大抑制所用到的nms_iou大小,越小越严格
}

# 实例化推理器
inference = TensorRTInfer(**config)

# 读取图片
IMAGE_PATH = r"./images/bus.jpg"
image_rgb = get_image(IMAGE_PATH)

# 单张图片推理
result, image_bgr_detect = inference.single(image_rgb, only_get_boxes=False)
print(result)
cv2.imshow("res", image_bgr_detect)
cv2.waitKey(0)

# 多张图片推理
IMAGE_DIR = r"../datasets/coco128/images/train2017"
SAVE_DIR  = r"../datasets/coco128/images/train2017_res"
# inference.multi(IMAGE_DIR, SAVE_DIR, save_xml=True) # save_xml 保存xml文件
```

# yolov5 export

```python
"""
Export a YOLOv5 PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit

Format                      | `export.py --include`         | Model
---                         | ---                           | ---
PyTorch                     | -                             | yolov5s.pt
TorchScript                 | `torchscript`                 | yolov5s.torchscript
ONNX                        | `onnx`                        | yolov5s.onnx
OpenVINO                    | `openvino`                    | yolov5s_openvino_model/
TensorRT                    | `engine`                      | yolov5s.engine
CoreML                      | `coreml`                      | yolov5s.mlmodel
TensorFlow SavedModel       | `saved_model`                 | yolov5s_saved_model/
TensorFlow GraphDef         | `pb`                          | yolov5s.pb
TensorFlow Lite             | `tflite`                      | yolov5s.tflite
TensorFlow Edge TPU         | `edgetpu`                     | yolov5s_edgetpu.tflite
TensorFlow.js               | `tfjs`                        | yolov5s_web_model/
PaddlePaddle                | `paddle`                      | yolov5s_paddle_model/

Requirements:
    $ pip install -r requirements.txt coremltools onnx onnxsim onnxruntime openvino-dev tensorflow-cpu  # CPU
    $ pip install -r requirements.txt coremltools onnx onnxsim onnxruntime-gpu openvino-dev tensorflow  # GPU

Usage:
    $ python export.py --weights yolov5s.pt --include torchscript onnx openvino engine coreml tflite ...

Inference:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolov5/yolov5s_web_model public/yolov5s_web_model
    $ npm start
"""

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--keras', action='store_true', help='TF: use Keras')
    parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
    parser.add_argument('--int8', action='store_true', help='CoreML/TF INT8 quantization')
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TF/TensorRT: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=17, help='ONNX: opset version')
    parser.add_argument('--verbose', action='store_true', help='TensorRT: verbose log')
    parser.add_argument('--workspace', type=int, default=4, help='TensorRT: workspace size (GB)')
    parser.add_argument('--nms', action='store_true', help='TF: add NMS to model')
    parser.add_argument('--agnostic-nms', action='store_true', help='TF: add agnostic NMS to model')
    parser.add_argument('--topk-per-class', type=int, default=100, help='TF.js NMS: topk per class to keep')
    parser.add_argument('--topk-all', type=int, default=100, help='TF.js NMS: topk for all classes to keep')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='TF.js NMS: IoU threshold')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='TF.js NMS: confidence threshold')
    parser.add_argument(
        '--include',
        nargs='+',
        default=['torchscript'],
        help='torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    print_args(vars(opt))
    return opt
```

> 导出路径和权重路径相同
>
> --include 后面写想导出的格式

## torchscript

```sh
python export.py --imgsz 640 --weights weights/yolov5s.pt --include torchscript --device 0
python export.py --imgsz 640 --weights weights/yolov5s.pt --include torchscript --device cpu --optimize # --optimize not compatible with cuda devices, i.e. use --device cpu
```

## onnx

> 注意:
>
> `onnxruntime` 和 `onnxruntime-gpu` 不要同时安装，否则使用 `gpu` 推理时速度会很慢，如果同时安装了2个包，要全部卸载，再安装 'onnxruntime-gpu' 才能使用gpu推理，否则gpu速度会很慢

```sh
python export.py --imgsz 640 --weights weights/yolov5s.pt --include onnx --simplify --device 0

python export.py --imgsz 640 --weights weights/yolov5s.pt --include onnx --simplify --device 0 --half      			# --half only compatible with GPU export, i.e. use --device 0

python export.py --imgsz 640 --weights weights/yolov5s.pt --include onnx --simplify --device cpu --dynamic 			# --dynamic only compatible with cpu

python export.py --imgsz 640 --weights weights/yolov5s.pt --include onnx --simplify --device cpu --half --dynamic	# 导出失败 --half not compatible with --dynamic
```

## openvino

```sh
python export.py --imgsz 640 --weights weights/yolov5s.pt --include openvino --simplify --device cpu      # 可以用simplify的onnx
python export.py --imgsz 640 --weights weights/yolov5s.pt --include openvino --simplify --device 0 --half # openvino支持half,但是要使用cpu导出onnx的half会报错,所以要使用 --device 0, openvino导出和设备无关,不受影响,主要是导出onnx的问题
```

### 通过openvino的`mo`命令将onnx转换为openvino格式(支持**fp16**)

> https://docs.openvino.ai/latest/notebooks/102-pytorch-onnx-to-openvino-with-output.html

```sh
mo --input_model "onnx_path" --output_dir "output_path" --compress_to_fp16

mo --input_model "onnx_path" --output_dir "output_path" --compress_to_fp16
```

#### 代码方式

```python
from openvino.tools import mo
from openvino.runtime import serialize

onnx_path = "onnx_path"

# fp32 IR model
fp32_path = "fp32_path"
output_path = fp32_path + ".xml"
print(f"Export ONNX to OpenVINO FP32 IR to: {output_path}")
model = mo.convert_model(onnx_path)
serialize(model, output_path)

# fp16 IR model
fp16_path = "fp16_path"
output_path = fp16_path + ".xml"

print(f"Export ONNX to OpenVINO FP16 IR to: {output_path}")
model = mo.convert_model(onnx_path, compress_to_fp16=True)
serialize(model, output_path)
```

### export failure  0.9s: DLL load failed while importing ie_api

> https://blog.csdn.net/qq_26815239/article/details/123047840
>
> 如果你使用的是 Python 3.8 或更高版本，并且是在Windows系统下通过pip安装的openvino，那么该错误的解决方案如下：

1. 进入目录 `your\env\site-packages\openvino\inference_engine`
2. 打开文件 `__init__.py`
3. 26行下添加一行

```python
        if os.path.isdir(lib_path):
            # On Windows, with Python >= 3.8, DLLs are no longer imported from the PATH.
            if (3, 8) <= sys.version_info:
                os.add_dll_directory(os.path.abspath(lib_path))
                os.environ['PATH'] = os.path.abspath(lib_path) + ';' + os.environ['PATH']	# 添加这一行
```

## tensorrt

```sh
python export.py --imgsz 640 --weights weights/yolov5s.pt --include engine --simplify --device 0 # 可以用simplify的onnx

python export.py --imgsz 640 --weights weights/yolov5s.pt --include engine --simplify --device 0 --half

python export.py --imgsz 640 --weights weights/yolov5s.pt --include engine --simplify --device 0 --dynamic --batch-size=16 	       # --dynamic model requires maximum --batch-size argument

python export.py --imgsz 640 --weights weights/yolov5s.pt --include engine --simplify --device 0 --half --dynamic --batch-size=16  # 导出失败 --half not compatible with --dynamic, i.e. use either --half or --dynamic but not both
```

## onnx openvino tensorrt

```sh
python export.py --imgsz 640 --weights weights/yolov5s.pt --include onnx openvino engine --simplify --device 0 --half 
```

