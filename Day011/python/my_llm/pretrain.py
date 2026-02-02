import argparse
import deepspeed
from torch.utils.tensorboard import SummaryWriter

from dataset import MyDataset
from model import *
from torch import nn

def parse_arguments():
    parser = argparse.ArgumentParser(description='Storier gpt train')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--data_file', default='./data/2022-05_zh_middle_0010',type=str)
    parser.add_argument('--ss',type=int)
    parser.add_argument('--batchsize',type=int)

    parser=deepspeed.add_config_arguments(parser)
    args=parser.parse_args()
    return args

class Trainer:
    def __init__(self):
        deepspeed.init_distributed()
        self.args = parse_arguments()

        rank=deepspeed.comm.get_rank()
        if rank==0:
            self.log=SummaryWriter("runs")

        self.model=Storier(num_layers=2,input_dim=128,hide_dim=96,n_q_heads=2,n_kv_heads=1,max_len=1024,num_vocs=50000,cache_max_batch_size=None,cache_max_seq_len=None)
        self.engine,self.opt,self.training_dataloader,self.lr_scheduler=deepspeed.initialize(
            args=self.args,
            model=self.model,
            training_data=MyDataset(f"{self.args.data_file}",self.args.batchsize),
            model_parameters=self.model.parameters(),
            config="./deepseed_config.json",
        )
        # 预训练阶段的损失函数忽略<pad>token对应的id=0的计算
        # 但在sft阶段不能忽略它
        self.loss_fn=nn.CrossEntropyLoss(ignore_index=0)

    def __call__(self):
        rank=deepspeed.comm.get_rank()
        self.engine.train()
        # deepspeed加载检查点的模型权重，默认加载最新权重
        # 如果想指定某一套权重，加 tag 参数
        _,client_sd=self.engine.load_checkpoint("weigths",tag=self.args.ss)#TODO 加载的模型地址,不加tag默认加载最新模型
        if client_sd is None:client_sd={"step":0}

        for _ in range(10000):
            for i,ds in enumerate(self.training_dataloader):
                ds=ds.to(device=self.engine.device,dtype=torch.long)
                xs=ds[:,:-1]#因果推理，模型输入，从第一个字符推后面所有字符
                ys=ds[:,1:]#模型输出
                os=self.engine(xs,0)

                os=os.reshape(-1,50000)#NSV
                os=os-os.mean(-1,keepdims=True)

                ys=ys.reshape(-1)#NS
                loss=self.loss_fn(os,ys)#自带onehot

                self.engine.backward(loss)
                self.engine.step()

                step=client_sd["step"]
                if rank==0:
                    if i%100==0:
                        self.log.add_scalar("loss",loss.item(),step)
                client_sd["step"]+=1

        save_tag=self.args.ss
        self.engine.save_checkpoint("weigths",tag=f"storier_{save_tag}",client_state={"step":client_sd['step']})#TODO 保存模型的地址,client_sd用以画tensorboard

if __name__ == '__main__':
    train=Trainer()
    train()