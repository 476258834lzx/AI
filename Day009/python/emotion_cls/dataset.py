from datasets import list_datasets,load_dataset,load_from_disk

# print(list_datasets())
# dataset=load_dataset(path="seamew/ChnSentiCorp",split="train")
dataset=load_from_disk("data/ChnSentiCorp")
# #取出训练集
train_data=dataset["train"]
# print(train_data)
#查看数据
for data in train_data:
    print(data)