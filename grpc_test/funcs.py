import os
import xml.etree.ElementTree as ET
import copy


def indent(elem, level=0):
    """缩进xml
    https://www.cnblogs.com/muffled/p/3462157.html
    """
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


with open("base.xml", mode="r", encoding="utf-8") as f:
    tree = ET.parse(f)


def json2xml(data: dict, path: str, file_name: str):
    """将检测的json转换为xml并保存

    Args:
        data (dict): json数据
        path (str): 保存路径
        file_name (str): 文件名
    """
    root = tree.getroot()
    # 获取临时object
    base_object = copy.deepcopy(root.find("object"))

    # 删除全部的object
    for o in root.findall("object"):
        root.remove(o)

    # 保存文件名
    root.find("filename").text = file_name + ".jpg"

    # 保存图片大小通道
    root.find("size").find('height').text = str(data["image_size"][0])
    root.find("size").find('width').text  = str(data["image_size"][1])
    root.find("size").find('depth').text  = str(data["image_size"][2])

    # 循环遍历保存框
    rectangles = data["detect"]
    for rectange in rectangles:
        # 需要重新copy,不然多个框只会保存最后一个
        temp_object = copy.deepcopy(base_object)
        # 保存类别名称和坐标
        temp_object.find("name").text = rectange["class"]

        temp_object.find("bndbox").find("xmin").text = str(rectange["box"][0])
        temp_object.find("bndbox").find("ymin").text = str(rectange["box"][1])
        temp_object.find("bndbox").find("xmax").text = str(rectange["box"][2])
        temp_object.find("bndbox").find("ymax").text = str(rectange["box"][3])

        # 将框保存起来
        root.append(temp_object)

    # 缩进root
    indent(root)
    new_tree = ET.ElementTree(root)
    xml_path = os.path.join(path, file_name+".xml")
    # 打开使用utf-8,写入时也需要utf-8
    new_tree.write(xml_path, encoding="utf-8")
