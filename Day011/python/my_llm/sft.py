import argparse
import deepspeed
import torch
from torch.utils.tensorboard import SummaryWriter
from data import *
from model import *
from torch import nn

def parse_arguments():
    parser = argparse.ArgumentParser(description="skyer SFT")
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--ss',default=0, type=int)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


class Trainer:

    def __init__(self):

        deepspeed.init_distributed()
        self.args = parse_arguments()

        rank = deepspeed.comm.get_rank()
        if rank == 0:
            self.log = SummaryWriter("runs")

        self.model = Storier(
            num_layers=48,
            input_dim=768,
            hide_dim=3072,
            n_q_heads=12,
            n_kv_heads=2,
            max_len=16384,
            num_vocs=50000,
            cache_max_batch_size=None,
            cache_max_seq_len=None
        )

        # weight = torch.load("xxx")["module"]
        # weight["_layer....w"] = torch.randn(....)
        
        # self.model.load_state_dict(weight)

        self.engine, self.opt, self.training_dataloader, self.lr_scheduler = deepspeed.initialize(
            args=self.args,
            model=self.model,
            training_data=SftDataset(f"{self.args.data_file}", 1024),
            model_parameters=self.model.parameters(),
            config="./deepspeed_config.json"
        )


        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def __call__(self):

        rank = deepspeed.comm.get_rank()

        self.engine.train()

        _, client_sd = self.engine.load_checkpoint(f"sft_save")
        if client_sd is None:
            client_sd = {"step": 0}

        for i, (prompt,tag) in enumerate(self.training_dataloader):
            xs = prompt[:, :-1].to(device=self.engine.device)
            ys = tag[:, 1:].to(device=self.engine.device)
            os = self.engine(xs)

            os = os.reshape(-1, 50000)
            os = os - os.mean(-1, keepdim=True)

            ys = ys.reshape(-1)

            loss = self.loss_fn(os, ys)

            self.engine.backward(loss)
            self.engine.step()

            step = client_sd['step']
            if rank == 0 and i % 100 == 0:
                self.log.add_scalar(f"loss", loss, step)
                
            client_sd['step'] += 1

        # hour = datetime.now().hour
        ss = self.args.ss
        self.engine.save_checkpoint(f"sft_save", tag=f"sft_{ss}",
                                    client_state={"step": client_sd['step']})


if __name__ == '__main__':
    train = Trainer()
    train()
