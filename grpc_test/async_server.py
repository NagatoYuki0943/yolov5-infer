import numpy as np
import cv2
import os
import time
import grpc
import pickle
from concurrent import futures
import base64
import trans_image_pb2
import trans_image_pb2_grpc
import asyncio
import sys
from funcs import json2xml
sys.path.append("../")
from onnxruntime_infer import OrtInference


SERVER_SAVE_PATH = "server"
os.makedirs(SERVER_SAVE_PATH, exist_ok=True)
SAVE = True # 是否保存图片和xml


class Server(trans_image_pb2_grpc.TransImageServicer):
    def __init__(self, inference) -> None:
        super().__init__()
        self.inference = inference

    async def trans(self, request: trans_image_pb2.DataRquest,
                    context: grpc.aio.ServicerContext)-> trans_image_pb2.DataResponse:
        """接收request,返回response
        trans是proto中service TransImage中的rpc trans
        """
        #=====================接收图片=====================#
        # 解码图片                               image是DataRquest中设定的变量
        image_decode = base64.b64decode(request.image)
        # 变成一个矩阵 单维向量
        array = np.frombuffer(image_decode, dtype=np.uint8)
        # print("array shape:", array.shape)
        # 再解码成图片 三维图片
        image_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
        print("image shape:", image_bgr.shape)

        #=====================预测图片=====================#
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_bgr_detect = self.inference.single(image_rgb)     # 推理返回绘制的图片
        detect: dict = self.inference.single_get_boxes(image_rgb)   # 推理返回框的数据,一般只需要一个推理即可
        detect["image_size"] = image_rgb.shape # 添加 [h, w, c]

        #================保存图片和检测结果=================#
        if SAVE:
            file_name = str(time.time())
            cv2.imwrite(os.path.join(SERVER_SAVE_PATH, file_name + ".jpg"), image_bgr)
            # 保存检测结果
            json2xml(detect, SERVER_SAVE_PATH, file_name)

        #=====================编码图片=====================#
        # 返回True和编码,这里只要编码
        image_encode = cv2.imencode(".jpg", image_bgr_detect)[1]
        # image_bytes = image_encode.tobytes()
        # image_64 = base64.b64encode(image_bytes)
        image_64 = base64.b64encode(image_encode)

        #=====================编码结果=====================#
        pickle_detect = pickle.dumps(detect)
        # 编码
        detect_64 = base64.b64encode(pickle_detect)

        #==================返回图片和结果===================#
        #                                   image和result是DataResponse中设定的变量
        return trans_image_pb2.DataResponse(image=image_64, detect=detect_64)


def get_inference():
    """获取推理器"""
    # 模型配置文件
    config = {
        'model_path':           "../weights/yolov5s.onnx",
        'mode':                 "cpu",
        'yaml_path':            "../weights/yolov5.yaml",
        'confidence_threshold': 0.25,   # 只有得分大于置信度的预测框会被保留下来,越大越严格
        'score_threshold':      0.2,    # nms分类得分阈值,越大越严格
        'nms_threshold':        0.45,   # 非极大抑制所用到的nms_iou大小,越小越严格
    }

    # 实例化推理器
    inference = OrtInference(**config)
    print("load inference!")
    return inference


async def run():
    # 最大客户端连接10(max_workers=10)，这里可定义最大接收和发送大小(单位M)，默认只有4M
    # 异步要用 aio
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[('grpc.max_send_message_length', 100 * 1024 * 1024),
                                  ('grpc.max_receive_message_length', 100 * 1024 * 1024)]
                            )
    # 绑定处理器
    trans_image_pb2_grpc.add_TransImageServicer_to_server(Server(get_inference()), server)

    # 绑定地址
    server.add_insecure_port("localhost:50054")
    await server.start()
    print('gRPC 服务端已开启，端口为50054...')
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(run())