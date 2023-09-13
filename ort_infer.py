from utils import read_image, OrtInference
import cv2


config = {
    "model_path":           r"./weights/yolov5s.onnx",
    "mode":                 r"cuda", # tensorrt cuda cpu
    "yaml_path":            r"./weights/yolov5.yaml",
    "confidence_threshold": 0.25,   # 只有得分大于置信度的预测框会被保留下来,越大越严格
    "score_threshold":      0.2,    # opencv nms分类得分阈值,越大越严格
    "nms_threshold":        0.6,    # 非极大抑制所用到的nms_iou大小,越小越严格
}

# 实例化推理器
inference  = OrtInference(**config)

# 读取图片
IMAGE_PATH = r"./images/bus.jpg"
image_rgb  = read_image(IMAGE_PATH)

# 单张图片推理
result, image_bgr_detect = inference.single(image_rgb, only_get_result=False, ignore_overlap_box=False)
print(result)
SAVE_PATH  = r"./ort_det.jpg"
cv2.imwrite(SAVE_PATH, image_bgr_detect)

# 多张图片推理
IMAGE_DIR  = r"../datasets/coco128/images/train2017"
SAVE_DIR   = r"../datasets/coco128/images/train2017_res"
# inference.multi(IMAGE_DIR, SAVE_DIR, save_xml=True) # save_xml 保存xml文件
# tensorrt: avg transform time: 3.625 ms, avg infer time: 24.0703125 ms, avg nms time: 0.0 ms, avg figure time: 12.3125 ms
# cuda:     avg transform time: 3.609375 ms, avg infer time: 22.3984375 ms, avg nms time: 0.0 ms, avg figure time: 12.078125 ms
# cpu:      avg transform time: 4.4296875 ms, avg infer time: 64.0625 ms, avg nms time: 0.015625 ms, avg figure time: 14.2265625 ms
