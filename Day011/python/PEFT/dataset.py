from modelscope.hub.snapshot_download import snapshot_download

snapshot_download("Qwen/Qwen2.5-0.5B-Instruct",cache_dir="./cache")

from modelscope.msdatasets import MsDataset
from modelscope  import AutoTokenizer

# ds=MsDataset.load("AI-ModelScope/ruozhiba",subset_name="post-annual",split=["train[:80%]","train[80%:]"])
ds=MsDataset.load("AI-ModelScope/ruozhiba",subset_name="post-annual",split={"train":"train[:80%]","test":"train[80%:]"})
_tokenizer=AutoTokenizer.from_pretrained("tiansz/bert-base-chinese")
_tokenizer.save_pretrained("./cache/tiansz-bert-base-chinese")

print(ds)
print(ds.column_names)
print(ds["train"].features)
print(ds["train"].train_test_split(0.1))

def handle(data,tokenizer=_tokenizer):
    return tokenizer(data["content"])

print(ds["train"]["content"][0])
_ds=ds["train"].map(handle,remove_columns=ds["train"].column_names,num_proc=2)
_ds=_ds.shuffle(seed=42).select(range(1000))
print(_ds[0])