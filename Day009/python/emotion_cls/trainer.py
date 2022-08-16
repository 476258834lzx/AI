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
train_dataset=MyDataset("train")
#创建dataloader
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True,drop_last=True,collate_fn=collate_function)

if __name__ == '__main__':
    # for i ,(input_ids,attention_mask,token_type_ids,labels) in enumerate(train_loader):
    #     print(i)
    #     print(input_ids.shape)
    #     print(attention_mask.shape)
    #     print(token_type_ids.shape)
    #     print(labels.shape)
    #开始训练
    model=Model().to(DEVICE)
    optimizer=AdamW(model.parameters(),lr=5e-4)
    loss_func=torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCH):
        for i ,(input_ids,attention_mask,token_type_ids,labels) in enumerate(train_loader):
            input_ids, attention_mask, token_type_ids, labels=input_ids.to(DEVICE),attention_mask.to(DEVICE),token_type_ids.to(DEVICE),labels.to(DEVICE)
            out=model(input_ids, attention_mask, token_type_ids)

            loss=loss_func(out,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%5==0:
                out=out.argmax(dim=1)
                acc=(out==labels).sum().item()/len(labels)
                print(epoch,i,loss.item(),acc)

        torch.save(model.state_dict(),f"params/{epoch}bert01.pth")
        print(epoch,"模型保存成功")