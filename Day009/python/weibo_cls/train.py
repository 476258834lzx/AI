import torch
from transformers import BertTokenizer,AdamW
from net import Model
from data import My_Dataset
from torch.utils.data import DataLoader

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH=30000

token=BertTokenizer.from_pretrained("yechen/bert-large-chinese")

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
train_dataset=My_Dataset("train")
val_dataset=My_Dataset("train")

#创建dataloader
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True,drop_last=True,collate_fn=collate_function)
val_loader=DataLoader(val_dataset,batch_size=32,shuffle=True,drop_last=True,collate_fn=collate_function)

if __name__ == '__main__':
    #开始训练
    print(DEVICE)
    model=Model().to(DEVICE)
    optimizer=AdamW(model.parameters())
    loss_func=torch.nn.CrossEntropyLoss()#自带标签one-hot编码

    model.train()
    for epo in range(EPOCH):
        #训练
        sum_val_loss=0
        sum_val_acc=0

        for i,(input_ids,attention_mask,token_type_ids,labels)in enumerate(train_loader):
            input_ids, attention_mask, token_type_ids, labels=input_ids.to(DEVICE),attention_mask.to(DEVICE),token_type_ids.to(DEVICE),labels.to(DEVICE)
            out=model(input_ids, attention_mask, token_type_ids)
            loss=loss_func(out,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%50==0:
                out=out.argmax(dim=1)
                acc=(out==labels).sum()/len(labels)
                print(epo,i,loss.item(),acc)

        #评估
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(
                val_loader):  # i*batchsize<=len(test_loader)
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), attention_mask.to(
                DEVICE), token_type_ids.to(DEVICE), labels.to(DEVICE)
            out = model(input_ids, attention_mask, token_type_ids)

            loss=loss_func(out,labels)

            out = out.argmax(dim=1)
            acc = (out == labels).sum().item()/len(labels)
            sum_val_loss+=loss
            sum_val_acc+=acc
        avg_val_loss=sum_val_loss/len(val_loader)
        avg_val_acc=sum_val_acc/len(val_loader)
        print(f"{epo}","avg_val_loss:",avg_val_loss,"avg_val_acc:",avg_val_acc)
        if epo%10==0:
            torch.save(model.state_dict(), f"params/{epo}-bert-weibo.pth")
            print(epo, "模型保存成功")