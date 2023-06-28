from datasets import load_dataset

# data=load_dataset(path="seamew/Weibo",split="train")
# print(data)
train_data=load_dataset(path="csv",data_files="data/train.csv",split="train")
print(train_data)
for data in train_data:
    print(data)
