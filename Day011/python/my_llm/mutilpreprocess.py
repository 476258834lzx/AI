import os
import json
import torch
from tqdm import tqdm
import sentencepiece as spm
import multiprocessing as mp


class Preprocessor:
    def __init__(self, filepath):
        self.spm = spm.SentencePieceProcessor()
        self.spm.Load("tokenizer.model")
        self.dst_path = filepath

    # 将核心逻辑提取为静态方法或独立函数，以便子进程使用
    # 注意：spm对象需要能在子进程中使用，或者在每个子进程初始化时重新加载
    # sentencepiece 的 Processor 对象通常可以pickle，但为了稳妥，建议在子进程中重新加载或在父进程加载后传递
    # 这里我们采用传递已加载的对象方式
    def process_file(self, file_path, lock):
        base_name = os.path.basename(file_path)
        filename = base_name.split('.')[0]
        vocs = []

        try:
            for line in open(file_path, "r+", encoding="utf-8"):
                try:
                    txt = json.loads(line)
                    ids = self.spm.Encode(txt["text"])
                    vocs.extend(ids)
                except json.JSONDecodeError:
                    continue

            vocs_tensor = torch.tensor(vocs, dtype=torch.uint16)

            # 使用锁保护写入过程
            with lock:
                # 虽然文件名不同，但加锁可以防止同一时刻大量进程并发写入同一磁盘目录可能引起的IO瓶颈或冲突
                torch.save(vocs_tensor, os.path.join(self.dst_path, filename))

            # 删除原文件也可以放在锁内，或者确保文件系统支持并发删除
            os.remove(file_path)
            return True

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False


# 包装函数，用于multiprocessing Pool
# 需要在全局作用域定义或者在 main 中定义
def worker_wrapper(args):
    preprocessor, file_path, lock = args
    return preprocessor.process_file(file_path, lock)


if __name__ == '__main__':
    # 确保多进程启动方式兼容性
    mp.set_start_method('spawn', force=True)

    preprocessor = Preprocessor("data")
    sky_path = "OpenDataLab___SkyPile-150B/raw/data"
    files = [os.path.join(sky_path, f) for f in os.listdir(sky_path)]

    # 创建管理器和锁
    manager = mp.Manager()
    lock = manager.Lock()

    # 设置进程数，建议根据CPU核心数调整
    num_processes = 4

    # 准备参数
    tasks = [(preprocessor, f, lock) for f in files]

    with mp.Pool(processes=num_processes) as pool:
        # 使用 imap_unordered 获得更好的性能，配合 tqdm 显示进度
        for _ in tqdm(pool.imap_unordered(worker_wrapper, tasks), total=len(files)):
            pass