"""onnx图片需要调整BGR2RGB, 并调整通道为[B, C, H, W], 且需要归一化
"""

import sys
sys.path.append("../")

import onnxruntime as ort
import numpy as np
from utils import Inference, check_onnx, load_yaml, single, multi


CONFIDENCE_THRESHOLD = 0.25 # 只有得分大于置信度的预测框会被保留下来,越大越严格
SCORE_THRESHOLD      = 0.2  # nms分类得分阈值,越大越严格
NMS_THRESHOLD        = 0.45 # 非极大抑制所用到的nms_iou大小,越小越严格


# print(ort.__version__)
print("onnxruntime all providers:", ort.get_all_providers())
print("onnxruntime available providers:", ort.get_available_providers())
# ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
print("ort devices:", ort.get_device())
# GPU


class OrtInference(Inference):
    def __init__(self, model_path: str, size: list[int], mode: str="cpu") -> None:
        """
        Args:
            model_path (str): 模型路径
            size (list[int]): 推理图片大小 [H, W]
            mode (str, optional): cpu cuda tensorrt. Defaults to cpu.
        """
        super().__init__()
        # 1.检测onnx模型
        check_onnx(model_path)
        # 2.保存图片宽高
        self.size = size
        # 3.载入模型
        self.model = self.get_model(model_path, mode)
        # 4.获取模型收入输出
        self.inputs = self.model.get_inputs()
        self.outputs = self.model.get_outputs()
        # 5.预热模型
        self.warm_up()

    def get_model(self, onnx_path: str, mode: str="cpu") -> ort.InferenceSession:
        """获取onnxruntime模型
        Args:
            onnx_path (str):      模型路径
            mode (str, optional): cpu cuda tensorrt. Defaults to cpu.
        Returns:
            ort.InferenceSession: 模型session
        """
        mode = mode.lower()
        assert mode in ["cpu", "cuda", "tensorrt"], "onnxruntime only support cpu, cuda and tensorrt inference."
        print(f"inference with {mode} !")

        so = ort.SessionOptions()
        so.log_severity_level = 3
        providers = {
            "cpu":  ['CPUExecutionProvider'],
            # https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
            "cuda": [
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024, # 2GB
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }),
                    'CPUExecutionProvider',
                ],
            # tensorrt
            # https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
            # it is recommended you also register CUDAExecutionProvider to allow Onnx Runtime to assign nodes to CUDA execution provider that TensorRT does not support.
            # set providers to ['TensorrtExecutionProvider', 'CUDAExecutionProvider'] with TensorrtExecutionProvider having the higher priority.
            "tensorrt": [
                    ('TensorrtExecutionProvider', {
                        'device_id': 0,
                        'trt_max_workspace_size': 2 * 1024 * 1024 * 1024, # 2GB
                        'trt_fp16_enable': False,
                    }),
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024, # 2GB
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    })
                ]
        }[mode]

        model = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)

        #--------------------------------#
        #   查看model中的内容
        #   get_inputs()返回对象，[0]返回名字
        #--------------------------------#
        # print("model outputs: \n", model.get_inputs())    # 列表 [<onnxruntime.capi.onnxruntime_pybind11_state.NodeArg object at 0x0000023BA140A770>]
        # print(model.get_inputs()[0])                      # NodeArg(name='images', type='tensor(float)', shape=[1, 3, 640, 640])
        # print(model.get_inputs()[0].name)                 # images
        # print(model.get_inputs()[0].type)                 # tensor(float)
        # print(model.get_inputs()[0].shape, "\n")          # [1, 3, 640, 640]

        # print("model outputs: \n", model.get_outputs())   # 列表 [<onnxruntime.capi.onnxruntime_pybind11_state.NodeArg object at 0x0000023BA140B5B0>]
        # print(model.get_outputs()[0])                     # NodeArg(name='output', type='tensor(float)', shape=[1, 25200, 85])
        # print(model.get_outputs()[0].name)                # output0
        # print(model.get_outputs()[0].type)                # tensor(float)
        # print(model.get_outputs()[0].shape, "\n")         # [1, 25200, 85]

        return model

    def warm_up(self):
        """预热模型
        """
        # [B, C, H, W]
        x = np.zeros((1, 3, *self.size), dtype=np.float32)
        self.infer(x)
        print("warmup finish")

    def infer(self, image: np.ndarray) -> list[np.ndarray]:
        """推理单张图片
        Args:
            image (np.ndarray): 图片 [B, C, H, W]
        Returns:
            np.ndarray: boxes [B, 25200, 85]
        """

        # 推理
        boxes = self.model.run(None, {self.inputs[0].name: image})    # 返回值为list

        return boxes


if __name__ == "__main__":
    YAML_PATH  = "../weights/yolov5.yaml"
    ONNX_PATH  = "../weights/yolov5s.onnx"
    IMAGE_PATH = "../images/bus.jpg"
    SAVE_PATH  = "./ort_det.jpg"

    # 获取label
    y = load_yaml(YAML_PATH)
    index2name = y["names"]
    # 实例化推理器
    inference = OrtInference(ONNX_PATH, y["size"], "cpu")
    # 单张图片推理
    single(inference, IMAGE_PATH, y["size"], index2name, CONFIDENCE_THRESHOLD, SCORE_THRESHOLD, NMS_THRESHOLD, SAVE_PATH)

    # 多张图片推理
    IMAGE_DIR = "../../datasets/coco128/images/train2017"
    SAVE_DIR  = "../../datasets/coco128/images/train2017_res"
    # multi(inference, IMAGE_DIR, y["size"], index2name, CONFIDENCE_THRESHOLD, SCORE_THRESHOLD, NMS_THRESHOLD, SAVE_DIR)
    # avg infer time: 74.71875 ms, avg nms time: 19.8984375 ms, avg figure time: 0.0 ms
