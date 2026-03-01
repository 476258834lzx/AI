import os
import json
import torch
from tqdm import tqdm
import sentencepiece as spm

class Preprocessor:
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
            vocs.extend(ids)
        vocs = torch.tensor(vocs,dtype=torch.uint16)
        torch.save(vocs, os.path.join(self.dst_path, filename))



if __name__ == '__main__':
    preprocessor = Preprocessor("data")

    sky_path="OpenDataLab___SkyPile-150B/raw/data"
    for file in tqdm(os.listdir(sky_path)):
        print(file)
        preprocessor(os.path.join(sky_path, file))
        os.remove(os.path.join(sky_path,file))