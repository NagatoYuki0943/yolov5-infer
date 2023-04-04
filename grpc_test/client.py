import numpy as np
import cv2
import os
import grpc
import base64
import pickle
import json
import object_detect_pb2
import object_detect_pb2_grpc



CLIENT_SAVE_PATH = "client"
os.makedirs(CLIENT_SAVE_PATH, exist_ok=True)


def run():
    """发送request,接收response
    """
    #=====================编码图片=====================#
    image = cv2.imread("../images/bus.jpg")
    # 返回True和编码,这里只要编码
    image_encode = cv2.imencode('.jpg', image)[1]
    # image_bytes = image_encode.tobytes()
    # image_64 = base64.b64encode(image_bytes)
    image_64 = base64.b64encode(image_encode)

    # 本次不使用SSL，所以channel是不安全的
    with grpc.insecure_channel("localhost:50054") as channel:
        # 客户端实例
        stub = object_detect_pb2_grpc.YoloDetectStub(channel)

        #=================发送并接收新图片==================#
        # V5Detect是proto中service YoloDetect中的rpc V5Detect
        #                                                  image是Request中设定的变量
        response = stub.V5Detect(object_detect_pb2.Request(image=image_64))

    # 解码图片                                image是Response中设定的变量
    image_decode = base64.b64decode(response.image)
    # 变成一个矩阵 单维向量
    array = np.frombuffer(image_decode, dtype=np.uint8)
    # 再解码成图片 三维图片
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    print(image.shape, image.dtype)
    cv2.imwrite(os.path.join(CLIENT_SAVE_PATH, "bus.jpg"), image)

    # 解码检测结果                             detect是Response中设定的变量
    detect_decode = base64.b64decode(response.detect)
    detect_res = pickle.loads(detect_decode)
    with open(os.path.join(CLIENT_SAVE_PATH, "detect.json"), mode="w", encoding="utf-8") as f:
        json.dump(detect_res, f, indent=4)
    print(detect_res)


if __name__ == "__main__":
    run()
