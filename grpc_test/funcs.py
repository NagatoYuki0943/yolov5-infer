from collections import Counter


def remap(data: dict, remap_dict: dict) -> dict:
    """将预测id转换为数据库id,删除不需要的数据

    Args:
        data (dict):       预测数据
        remap_dict (dict): origin id to database id dict

    Returns:
        dict:              remap的数据
    """
    new_count = []
    for box in data["detect"]:
        # 去除名字
        box.pop("class")
        # 映射id
        box["class_index"] = remap_dict[box["class_index"]]

        # 忽略new_id为0的类别,这个类别不要
        if box["class_index"] == 0:
            data["detect"].remove(box)
            continue

        new_count.append(box["class_index"])

    # reamp id 计数
    data["count"] = dict(Counter(new_count))

    return data
