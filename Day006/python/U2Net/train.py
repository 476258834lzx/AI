import torch.nn
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from data import Mydataset
from net import *
from torch import optim
from torchvision.utils import save_image
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
    def __init__(self, img_dir, weights_dir, result_path,local_rank):
        # 初始化
        self.summaryWriter = SummaryWriter("logs")
        self.result_path = result_path
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")#初始化DDP
        self.local_rank = local_rank
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", local_rank)

        # 加载训练数据
        self.dataset = Mydataset(img_dir)
        self.sampler=DistributedSampler(self.dataset)
        # self.dataloader = DataLoader(self.dataset, 4, sampler=self.sampler)
        self.dataloader = DataLoader(self.dataset, 4, True, drop_last=True, num_workers=1,sampler=self.sampler)

        # 加载模型
        self.net = U2NET().to(self.device)

        # 加载预训练权重
        self.weights_dir = weights_dir
        if dist.get_rank() == 0:
            if os.path.exists(os.path.join(self.weights_dir, "last.pt")):
                self.net.load_state_dict(torch.load(os.path.join(self.weights_dir, "last.pt")))

        # 转DDP模型
        self.model=DDP(self.net,device_ids=[self.local_rank], output_device=self.local_rank)

        # 优化算法
        self.opt = optim.Adam(self.net.parameters())

        # 损失函数
        self.loss_function = SegmentationLosses(criterion_weight=torch.Tensor([0.035225969, 0.072890643, 0.389344428])).to(self.device)

    def __call__(self):
        epoch=0
        while True:
            for i,(xs,ys) in enumerate(self.dataloader):
                self.model.train()
                self.dataloader.sampler.set_epoch(epoch)

                xs,ys=xs.to(self.device),ys.long().to(self.device)
                d0,d1,d2,d3,d4,d5,d6=self.model(xs)
                loss0=self.loss_function(d0,ys)
                loss1=self.loss_function(d1,ys)
                loss2=self.loss_function(d2,ys)
                loss3=self.loss_function(d3,ys)
                loss4=self.loss_function(d4,ys)
                loss5=self.loss_function(d5,ys)
                loss6=self.loss_function(d6,ys)

                loss=loss0+loss1+loss2+loss3+loss4+loss5+loss6
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                print(loss)
                if epoch%2==0:
                    self.summaryWriter.add_scalars("loss", {"train_loss": loss}, i)

            epoch+=1
            if epoch%10==0 and dist.get_rank() == 0:
                torch.save(self.model.module.state_dict(), f"{self.weights_dir}/{epoch}.pt")


if __name__ == '__main__':
    train = Train("data","weights","result","0,1")
    train()