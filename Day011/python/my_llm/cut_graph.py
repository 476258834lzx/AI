import os
import pickle
import numpy as np
from collections import defaultdict
from pathlib import Path


def get_padded_length(original_length, base=256, max_length=10240):
    """
    计算padding后的长度：大于自身长度的第一个256的整数倍，但不超过10240
    """
    if original_length > max_length:
        return max_length

    # 计算大于original_length的第一个base的倍数
    padded = ((original_length + base - 1) // base) * base
    return min(padded, max_length)


def process_ndarray(arr):
    """
    处理单个ndarray：padding或截断
    """
    original_length = len(arr)
    target_length = get_padded_length(original_length)

    if original_length > 10240:
        # 截断到10240，并将最后一个元素修改为3
        result = arr[:10240].copy()
        result[-1] = 3
        return result
    else:
        # 需要padding到target_length
        if original_length < target_length:
            # 创建新数组并填充0
            result = np.zeros(target_length, dtype=arr.dtype)
            result[:original_length] = arr
            return result
        else:
            # 正好等于target_length，直接返回
            return arr


def save_batch(data_dict, output_dir, length_key, batch_id):
    """
    保存一批数据到文件
    """
    filename = f"{length_key}_{batch_id}.token"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'wb') as f:
        pickle.dump(data_dict[length_key], f)

    print(f"Saved: {filename} (count: {len(data_dict[length_key])})")
    # 清空该长度的列表
    data_dict[length_key] = []


def process_token_files(input_dir, output_dir):
    """
    主处理函数
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 数据结构：按长度分组的字典，value是列表
    # 格式：{长度: [ndarray1, ndarray2, ...]}
    length_groups = defaultdict(list)

    # 记录每个长度对应的当前batch id
    batch_counters = defaultdict(int)

    # 每个list的最大长度
    MAX_LIST_SIZE = 5000000

    # 获取所有.token文件
    token_files = list(Path(input_dir).glob("*.token"))
    print(f"Found {len(token_files)} .token files")

    total_processed = 0

    for file_path in token_files:
        print(f"Processing: {file_path.name}")

        # 读取pickle文件
        with open(file_path, 'rb') as f:
            data_list = pickle.load(f)

        # 确保是list
        if not isinstance(data_list, list):
            print(f"Warning: {file_path.name} does not contain a list, skipping")
            continue

        # 处理list中的每个ndarray
        for arr in data_list:
            if not isinstance(arr, np.ndarray):
                print(f"Warning: non-ndarray item found, type: {type(arr)}, skipping")
                continue

            # 确保是一维
            arr = np.asarray(arr).flatten()

            # 处理array（padding或截断）
            processed_arr = process_ndarray(arr)
            arr_length = len(processed_arr)

            # 添加到对应长度的组
            length_groups[arr_length].append(processed_arr)
            total_processed += 1

            # 检查是否达到最大长度，需要保存
            if len(length_groups[arr_length]) >= MAX_LIST_SIZE:
                save_batch(length_groups, output_dir, arr_length, batch_counters[arr_length])
                batch_counters[arr_length] += 1

        print(f"  Processed {len(data_list)} arrays from {file_path.name}")

    # 保存剩余的数据
    print("\nSaving remaining data...")
    for length_key, arr_list in length_groups.items():
        if len(arr_list) > 0:
            filename = f"{length_key}_{batch_counters[length_key]}.token"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'wb') as f:
                pickle.dump(arr_list, f)

            print(f"Saved: {filename} (count: {len(arr_list)})")

    print(f"\nDone! Total processed arrays: {total_processed}")

    # 打印统计信息
    print("\nStatistics:")
    for length_key in sorted(length_groups.keys()):
        batch_count = batch_counters[length_key] + (1 if len(length_groups[length_key]) > 0 else 0)
        total_count = batch_counters[length_key] * MAX_LIST_SIZE + len(length_groups[length_key])
        print(f"  Length {length_key}: {total_count} arrays in {batch_count} file(s)")


if __name__ == "__main__":
    INPUT_DIRECTORY = "tmp"  # 输入文件夹
    OUTPUT_DIRECTORY = "data"  # 输出文件夹

    process_token_files(INPUT_DIRECTORY, OUTPUT_DIRECTORY)