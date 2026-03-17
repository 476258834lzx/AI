import torch
import torch.nn as nn
import torch.nn.functional as F

class IPOLoss(nn.Module):
    """
    Identity Preference Optimization (IPO) Loss
    参考论文: "A General Theoretical Paradigm to Understand Learning from Human Preferences"
    https://arxiv.org/abs/2310.12036
    """
    def __init__(self, beta: float = 0.1, τ: float = 0.1):
        super().__init__()
        self.beta = beta  # 温度参数，控制偏离参考模型的KL惩罚强度
        self.τ = τ    # IPO特定参数，控制对偏好差异的容忍度。目标差值约为 1/(2*τ)

    def _get_batch_logps(
        self,
        logits: torch.Tensor,        # [batch_size, seq_len, vocab_size]
        labels: torch.Tensor,         # [batch_size, seq_len]
        attention_mask: torch.Tensor  # [batch_size, seq_len], 1 for valid tokens
    ) -> torch.Tensor:
        """
        计算每个序列在有效token上的平均log probability (或求和，根据习惯)
        这里采用求和，但用mask确保只计算有效token。
        """
        # 计算log softmax
        log_probs = F.log_softmax(logits, dim=-1)
        # 获取每个token的log prob
        per_token_logps = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        # 应用mask，将padding位置的log prob置为0，然后求和
        per_token_logps = per_token_logps * attention_mask
        # 对每个序列的有效token求和
        return per_token_logps.sum(dim=-1)

    def forward(
        self,
        policy_chosen_logits: torch.Tensor,
        policy_rejected_logits: torch.Tensor,
        reference_chosen_logits: torch.Tensor,
        reference_rejected_logits: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_labels: torch.Tensor,
        chosen_attention_mask: torch.Tensor,   # 必须传入mask
        rejected_attention_mask: torch.Tensor, # 必须传入mask
        reduce: str = 'mean'                    # 控制返回形式
    ) -> torch.Tensor:
        """
        计算IPO Loss
        Args:
            reduce: 'mean' 返回batch平均损失, 'none' 返回每个样本的损失
        """
        # 1. 计算带mask的对数概率
        policy_chosen_logps = self._get_batch_logps(policy_chosen_logits, chosen_labels, chosen_attention_mask)
        policy_rejected_logps = self._get_batch_logps(policy_rejected_logits, rejected_labels, rejected_attention_mask)
        reference_chosen_logps = self._get_batch_logps(reference_chosen_logits, chosen_labels, chosen_attention_mask)
        reference_rejected_logps = self._get_batch_logps(reference_rejected_logits, rejected_labels, rejected_attention_mask)

        # 2. 计算对数概率比
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        # 3. Beta缩放
        scaled_chosen = chosen_logratios / self.beta
        scaled_rejected = rejected_logratios / self.beta

        # 4. 计算核心logits（偏好差值）
        pi_pref_logits = scaled_chosen - scaled_rejected

        # 5. IPO平方损失
        target = 1.0 / (2 * self.τ)
        losses = (pi_pref_logits - target) ** 2

        if reduce == 'mean':
            return losses.mean()
        elif reduce == 'none':
            return losses
        else:
            raise ValueError(f"Invalid reduce option: {reduce}")