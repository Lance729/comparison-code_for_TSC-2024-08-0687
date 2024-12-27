import json
from pathlib import Path
import numpy as np

def save_data_to_json(file_path: str, new_data: dict, insert_at: str = None, top_level_key: str = "data"):
    """
    将数据保存到 JSON 文件中。可以指定插入数据的键位置，若未指定位置，则将数据添加到文件末尾。

    参数：
    - file_path (str): 保存数据的目标文件路径。
    - new_data (dict): 需要保存的数据，应该是一个字典类型的数据。
    - insert_at (str, 可选): 指定插入的数据键。默认为 None，表示数据将被添加到文件的末尾。
    - top_level_key (str): 传入一个顶层键名，避免使用 "null" 作为默认键名，默认为 "data"。
    
    功能：
    1. 如果文件不存在，则创建新的 JSON 文件并保存数据。
    2. 如果文件已经存在，将新数据插入到指定位置（如果提供了位置），否则将新数据追加到文件的末尾。
    3. 数据保存时以 JSON 格式存储，保持原有文件内容的格式。
    """

    # 创建文件路径对象
    path = Path(file_path)

    # 如果文件不存在，直接创建并写入新数据
    if not path.exists():
        with open(path, 'w', encoding='utf-8') as json_file:
            # 用明确的顶层键名（例如 "data"）保存数据
            json.dump({top_level_key: new_data}, json_file, ensure_ascii=False, indent=4)
        print(f"文件 '{file_path}' 不存在，已创建并保存数据。")
        return

    # 如果文件存在，先读取现有的数据
    with open(path, 'r', encoding='utf-8') as json_file:
        existing_data = json.load(json_file)

    # 检查现有数据是否是字典
    if isinstance(existing_data, dict):
        if insert_at is None:
            # 如果未指定插入位置，将新数据插入到字典中
            existing_data.update(new_data)
        else:
            # 如果指定了插入位置，将新数据作为新的键值对添加到指定位置
            existing_data[insert_at] = new_data
    else:
        raise TypeError("现有数据不是一个字典结构，无法进行插入操作。")

    # 保存更新后的数据回文件
    with open(path, 'w', encoding='utf-8') as json_file:
        json.dump(existing_data, json_file, ensure_ascii=False, indent=4)

    print(f"数据已成功保存到 '{file_path}'，并且数据已插入到位置 {insert_at if insert_at is not None else '末尾'}。")






def read_json_file(file_path):
    """读取JSON文件并返回数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return None




# def convert_to_row_vector():
#     # 读取 JSON 文件
#     with open('results/actors_copy.json', 'r', encoding='utf-8') as file:
#         data = json.load(file)

#     # 遍历 JSON 文件中的每个数组，并将它们转换为行向量
#     for scheme in data['schemes']:
#         for actor in data['schemes'][scheme]:
#             # 将列向量转换为行向量
#             data['schemes'][scheme][actor] = [data['schemes'][scheme][actor]]

#     # 将修改后的数据写入 JSON 文件
#     with open('results/actors_copy.json', 'w', encoding='utf-8') as file:
#         json.dump(data, file, ensure_ascii=False, indent=4)



# if __name__ == '__main__':
    # # 示例用法
    # file_path = './results/actors.json'
    # data = read_json_file(file_path)
    # if data is not None:
    #     print(data['actor_schemes'])
    # convert_to_row_vector()