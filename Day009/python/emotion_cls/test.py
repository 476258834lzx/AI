import torch
from data import MyDataset
from torch.utils.data import DataLoader
from net import Model
from transformers import AdamW,BertTokenizer

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH=30000

token=BertTokenizer.from_pretrained("bert-base-chinese")

def collate_function(data):
    sentes=[i[0] for i in data]
    labels=[i[1] for i in data]
    #编码
    data=token.batch_encode_plus(batch_text_or_text_pairs=sentes,truncation=True,padding="max_length",max_length=500,return_tensors="pt",return_length=True)
    input_ids=data['input_ids']
    attention_mask=data['attention_mask']
    token_type_ids=data['token_type_ids']
    labels=torch.LongTensor(labels)
    return input_ids,attention_mask,token_type_ids,labels

#创建数据集
test_dataset=MyDataset("test")
#创建dataloader
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=True,drop_last=True,collate_fn=collate_function)

if __name__ == '__main__':
    #开始测试
    print(DEVICE)
    model=Model().to(DEVICE)
