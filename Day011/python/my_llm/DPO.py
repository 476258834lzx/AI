from torch import nn
import torch
import torch.nn.functional as F


class DPOLoss(nn.Module):
    def __init__(self, beta: float, label_smoothing: float = 0.0):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def _get_batch_logps(self, logits, labels):
        """计算每个序列的平均 log probability"""
        # logits: [batch, seq_len, vocab_size]
        # labels: [batch, seq_len]

        log_probs = F.log_softmax(logits, dim=-1)

        # 收集目标 token 的 log prob
        per_token_logps = torch.gather(
            log_probs,
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)

        # 计算每个序列的有效长度（忽略 padding，假设 pad_token_id=0 或 -100）
        loss_mask = (labels != -100).float()

        # 序列级别的平均 log prob
        sequence_logps = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)

        return sequence_logps

    def forward(self,
                policy_chosen_logps: torch.Tensor,  # [batch]
                policy_rejected_logps: torch.Tensor,  # [batch]
                reference_chosen_logps: torch.Tensor,  # [batch]
                reference_rejected_logps: torch.Tensor):  # [batch]
        """
        输入应该是已经计算好的序列级 log probabilities
        或者如果输入是 logits，先调用 _get_batch_logps 转换
        """

        # Policy 和 Reference 的 log ratio
        policy_logratios = policy_chosen_logps - policy_rejected_logps
        reference_logratios = reference_chosen_logps - reference_rejected_logps

        # DPO 的 logits（reward margin）
        logits = self.beta * (policy_logratios - reference_logratios)

        # 数值稳定的实现: -log(sigmoid(logits)) = softplus(-logits)
        losses = F.softplus(-logits)

        if self.label_smoothing > 0:
            # Label smoothing: 混合 -log(sigmoid(logits)) 和 -log(sigmoid(-logits))
            losses = (1 - self.label_smoothing) * losses + self.label_smoothing * F.softplus(logits)

        return losses.mean()