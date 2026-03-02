import os
import json
import torch
import pickle
from tqdm import tqdm
import sentencepiece as spm

class SkyPilePreprocessor:
    def __init__(self,filepath):
        self.spm = spm.SentencePieceProcessor()
        self.spm.Load("tokenizer.model")
        self.dst_path=filepath

    def __call__(self, file_path):
        base_name=os.path.basename(file_path)
        filename=base_name.split('.')[0]
        vocs=[]
        for line in open(file_path, "r+", encoding="utf-8"):
            txt = json.loads(line)
            ids = self.spm.Encode(txt["text"])
            vocs.append(2)
            vocs.extend(ids)
            vocs.append(3)
        vocs = torch.tensor(vocs,dtype=torch.uint16)
        torch.save(vocs, os.path.join(self.dst_path, filename))

class SftPreprocessor:
    def __init__(self, filepath):
        self.spm = spm.SentencePieceProcessor()
        self.spm.Load("tokenizer.model")
        self.dst_path = filepath

    def __call__(self, file_path):
        base_name = os.path.basename(file_path)
        filename = base_name.split('.')[0]
        datas = []

        with open(file_path, "r+", encoding="UTF-8") as fr:
            js = json.load(fr)
            for obj in js:
                user = obj["instruction"]
                assistant = obj["output"]

                input = f"<|im_start|>system\n以下的问题用中文回答。<|im_end|><|im_start|>user\n{user}<|im_end|><|im_start|>assistant\n"
                output = f"{assistant}<|im_end|>"

                input_ids = self.spm.Encode(input)
                output_ids = self.spm.Encode(output)

                prompt = input_ids + output_ids
                tag = len(input) * [0, ] + output_ids

                datas.append([prompt, tag])

        with open(f"{self.dst_path}/{filename}.bin", "wb") as fw:
            pickle.dump(datas, fw)


if __name__ == '__main__':
    preprocessor = SkyPilePreprocessor("data")

    sky_path="OpenDataLab___SkyPile-150B/raw/data"
    for file in tqdm(os.listdir(sky_path)):
        print(file)
        preprocessor(os.path.join(sky_path, file))
        os.remove(os.path.join(sky_path,file))