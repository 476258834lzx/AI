import torch.optim as optim
from tensorboard.plugins.scalar.summary import scalar
from torch.utils.data import DataLoader
from data import Mydataset
from net import *
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import os
from torch import multiprocessing as mp
from torch.amp import autocast as autocast, GradScaler as GradScaler

class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes=25, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
        self.weight = weight
                     #N*3*H*W  N*H*W
    def forward(self, logits, targets):
        preds = F.softmax(logits, dim=1)
        num_classes = preds.shape[1]
        target_clamp = torch.clamp(targets.long(), 0, num_classes - 1)

        one_hot_target = F.one_hot(
            target_clamp,
            num_classes=num_classes)#N*H*W*3
        total_loss = 0.0
        smooth = 1
        for i in range(num_classes):
            pred = preds[:, i]#N*H*W
            target = one_hot_target[..., i]

            pred = pred.reshape(pred.shape[0], -1)#展开NV
            target = target.reshape(target.shape[0], -1)#NV

            num = torch.sum(torch.mul(pred, target), dim=1) * 2 + smooth
            den = torch.sum(pred.pow(2) + target.pow(2), dim=1) + smooth

            dice_loss = 1 - num / den
            if self.weight is not None:
                dice_loss *= self.weight[i]
            total_loss += dice_loss

        total_loss = total_loss / num_classes
        return total_loss.sum()

class SegFocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None,ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,reduction='none')

    def forward(self, input, target):
        # target=torch.maximum(target,torch.tensor([1e-6]).long().cuda())
        logpt = -self.ce_fn(input, target)
        logpt = torch.minimum(logpt, torch.tensor(3.5))
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        loss_=loss.mean()
        return loss_

class SegmentationLosses(nn.Module):
    def __init__(self,criterion_weight=None, ignore_lb=255):
        super(SegmentationLosses, self).__init__()
        self.ignore_lb = ignore_lb
        self.focal = SegFocalLoss(weight=criterion_weight)
        self.dice = SoftDiceLoss(weight=criterion_weight)
        # self.abl=ABL()

    def forward(self, logits, targets):
        # targets[targets<0]=float(1e-6)
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        # abl_loss=self.abl(logits, targets)
        # print(focal_loss.item())
        # print(dice_loss.item())

        return focal_loss+dice_loss

class Train:
    def __init__(self, rank, world_size,img_dir,weights_dir):
        self.summaryWriter = SummaryWriter("logs")
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")
        self.batch_size = 2
        self.lr = 0.002
        self.num_epochs = 10000000
        self.scaler = GradScaler()

        # 初始化分布式训练
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:12345",
            rank=rank,
            world_size=world_size
        )

        # 加载数据
        train_dataset = Mydataset(img_dir)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True, sampler=train_sampler)

        # 初始化模型和优化器
        self.model = U2NET().to(self.device)

        self.weights_dir = weights_dir
        if dist.get_rank() == 0:
            if os.path.exists(os.path.join(self.weights_dir, "last.pt")):
                self.model.load_state_dict(torch.load(os.path.join(self.weights_dir, "last.pt")))

        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = DDP(self.model, device_ids=[self.rank],find_unused_parameters=True, bucket_cap_mb=25)
        self.criterion = SegmentationLosses(criterion_weight=torch.Tensor([0.035225969, 0.072890643, 0.389344428])).to(self.device)
        self.optimizer = optim.NAdam(self.model.parameters(), lr=self.lr)

    def train(self):
        for epoch in range(self.num_epochs):
            self.train_loader.sampler.set_epoch(epoch)
            sum_loss=0
            account=len(self.train_loader)
            for i, (data, target) in enumerate(self.train_loader):
                self.model.train()
                data, target = data.to(self.device), target.long().to(self.device)

                self.optimizer.zero_grad()
                with autocast():
                    d0,d1,d2,d3,d4,d5,d6 = self.model(data)
                    loss0 = self.criterion(d0, target)
                    loss1 = self.criterion(d1, target)
                    loss2 = self.criterion(d2, target)
                    loss3 = self.criterion(d3, target)
                    loss4 = self.criterion(d4, target)
                    loss5 = self.criterion(d5, target)
                    loss6 = self.criterion(d6, target)
                    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                self.scaler(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                sum_loss+=loss.item()
                if i % 10 == 0:
                    print("Rank %d, Epoch %d, Batch %d, Loss %.4f" % (self.rank, epoch, i, loss.item()))
            if epoch%1==0 and dist.get_rank() == 0:
                torch.save(self.model.module.state_dict(), f"{self.weights_dir}/{epoch}.pt")
                self.summaryWriter.add_scalars("loss", {"train_loss": sum_loss/account}, epoch)
        torch.cuda.empty_cache()

def run(rank, world_size):
    train = Train(rank,world_size,"data","weights")
    train.train()

if __name__ == "__main__":
    num_gpus = 4
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12345"
    mp.spawn(run, args=(num_gpus,), nprocs=num_gpus, join=True)
