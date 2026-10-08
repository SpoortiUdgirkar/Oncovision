"""
Custom Loss Functions for Model Training & Optimization.

Contains:
- FocalLoss: Multi-class Focal Loss (Lin et al., 2017) to dynamically down-weight
  easy examples and focus training on hard negative/positive samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss (Lin et al., 2017).

    Formula:
        FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
        where p_t is the model's estimated probability for the ground-truth class t.

    Args:
        gamma (float): Focusing parameter gamma >= 0. Default: 2.0.
        alpha (Tensor, list, or tuple, optional): Class weights of shape (C,). Default: None (unweighted).
        reduction (str): 'mean' | 'sum' | 'none'. Default: 'mean'.
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Forward pass for multiclass focal loss.

        Args:
            inputs (torch.Tensor): Model logits of shape (N, C)
            targets (torch.Tensor): Ground-truth class indices of shape (N,)

        Returns:
            torch.Tensor: Computed loss value
        """
        if inputs.dim() != 2:
            raise ValueError(f"FocalLoss expects 2D inputs of shape (N, C), got shape {inputs.shape}")
        if targets.dim() != 1:
            raise ValueError(f"FocalLoss expects 1D targets of shape (N,), got shape {targets.shape}")

        # Compute log probabilities and probabilities in numerically stable way
        log_p = F.log_softmax(inputs, dim=1)  # (N, C)
        p = torch.exp(log_p)                  # (N, C)

        # Extract log probability and probability for the ground-truth class t
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)          # (N,)

        # Compute focal modulating factor (1 - pt)^gamma
        focal_weight = (1.0 - pt) ** self.gamma

        # Unweighted focal loss per sample
        loss = -focal_weight * log_pt  # (N,)

        # Optional class weighting alpha_t
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
