from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class A1Head(nn.Module):

    def __init__(self, d_in: int, bias_init: list[float] | None = None) -> None:
        super().__init__()
        self.fc = nn.Linear(d_in, 3)
        if bias_init is not None:
            with torch.no_grad():
                self.fc.bias.copy_(torch.tensor(bias_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    @staticmethod
    def predict_probs(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)


def a2_logits_to_monotonic_class_probs(logits: torch.Tensor) -> torch.Tensor:
    """Map CORAL/Cumulative logits to a proper simplex (P0..P3), differentiable."""
    s = torch.sigmoid(logits)
    p1 = s[..., 0]
    p2 = torch.min(s[..., 1], p1)
    p3 = torch.min(s[..., 2], p2)
    p0 = 1.0 - p1
    c1 = p1 - p2
    c2 = p2 - p3
    c3 = p3
    probs = torch.stack([p0, c1, c2, c3], dim=-1)
    probs = probs.clamp_min(0.0)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class A2OrdinalHead(nn.Module):
    def __init__(self, d_in: int, n_items: int = 21, n_thresholds: int = 3) -> None:
        super().__init__()
        self.n_items = n_items
        self.n_thresholds = n_thresholds
        self.fc = nn.Linear(d_in, n_items * n_thresholds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        return self.fc(x).view(B, self.n_items, self.n_thresholds)

    @staticmethod
    def predict_int(logits: torch.Tensor) -> torch.Tensor:
        return (torch.sigmoid(logits) > 0.5).long().sum(dim=-1)

    @staticmethod
    def predict_int_monotonic(logits: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(logits)  

        p1 = s[..., 0]
        p2 = torch.min(s[..., 1], p1)
        p3 = torch.min(s[..., 2], p2)

        P0 = 1.0 - p1
        P1 = p1 - p2
        P2 = p2 - p3
        P3 = p3
        class_probs = torch.stack([P0, P1, P2, P3], dim=-1)  
        return class_probs.argmax(dim=-1) 

    @staticmethod
    def predict_expectation(logits: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(logits)
        p1 = s[..., 0]
        p2 = torch.min(s[..., 1], p1)
        p3 = torch.min(s[..., 2], p2)
        E = p1 + p2 + p3  
        return E.round().long().clamp(0, 3)

    @staticmethod
    def build_ordinal_targets(labels: torch.Tensor, n_thresholds: int = 3) -> torch.Tensor:
        B, I = labels.shape
        thresholds = torch.arange(1, n_thresholds + 1, device=labels.device).float()
        targets = (labels.unsqueeze(-1).float() >= thresholds.view(1, 1, -1)).float()
        return targets



def a1_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if label_smoothing > 0.0:
        targets = targets.float() * (1.0 - label_smoothing) + 0.5 * label_smoothing
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def a2_ordinal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    targets = A2OrdinalHead.build_ordinal_targets(labels, n_thresholds=logits.size(-1))
    if label_smoothing > 0.0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def _soft_expected_ordinal_score(logits: torch.Tensor) -> torch.Tensor:
    """E[Y] for ordinal labels 0..K with cumulative logits: E[Y] = sum_k sigmoid(z_k)."""
    return torch.sigmoid(logits).sum(dim=-1)


def a2_soft_qwk_loss_from_probs(
    class_probs: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Differentiable soft QWK: mean (1 - kappa) over items."""
    b_sz, _n_items, k = class_probs.shape
    device = class_probs.device
    dtype = class_probs.dtype
    idx = torch.arange(k, device=device, dtype=dtype)
    w = (idx.unsqueeze(0) - idx.unsqueeze(1)).pow(2) / ((k - 1) ** 2)

    true_oh = F.one_hot(labels.long(), k).to(dtype)
    o = torch.einsum("bit,bij->itj", true_oh, class_probs) / float(b_sz)
    hist_true = true_oh.mean(dim=0)
    hist_pred = class_probs.mean(dim=0)
    e_mat = hist_true.unsqueeze(2) * hist_pred.unsqueeze(1)
    num = (w.unsqueeze(0) * o).sum(dim=(1, 2))
    den = (w.unsqueeze(0) * e_mat).sum(dim=(1, 2)).clamp_min(eps)
    kappa = 1.0 - num / den
    return (1.0 - kappa).mean()


def a2_ordinal_emd_loss_from_probs(class_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """1D Wasserstein-1 / ordinal earth mover distance between pred distribution and one-hot labels."""
    k = class_probs.size(-1)
    true_oh = F.one_hot(labels.long(), k).to(class_probs.dtype)
    diff = class_probs - true_oh
    cdf_diff = torch.cumsum(diff, dim=-1)
    emd = cdf_diff[..., :-1].abs().sum(dim=-1)
    return emd.mean()


def a2_combined_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    coral_weight: float = 1.0,
    soft_qwk_weight: float = 0.0,
    emd_weight: float = 0.0,
    expectation_weight: float = 0.0,
    expectation_distance_aware: bool = False,
    expectation_distance_power: float = 1.0,
) -> torch.Tensor:
    coral = a2_ordinal_loss(logits, labels, pos_weight=pos_weight, label_smoothing=label_smoothing)
    total = coral_weight * coral
    if soft_qwk_weight <= 0.0 and emd_weight <= 0.0 and expectation_weight <= 0.0:
        return total
    probs = a2_logits_to_monotonic_class_probs(logits)
    if soft_qwk_weight > 0.0:
        total = total + soft_qwk_weight * a2_soft_qwk_loss_from_probs(probs, labels)
    if emd_weight > 0.0:
        total = total + emd_weight * a2_ordinal_emd_loss_from_probs(probs, labels)
    if expectation_weight > 0.0:
        exp_score = _soft_expected_ordinal_score(logits)
        exp_loss = F.smooth_l1_loss(exp_score, labels.float(), reduction="none")
        if expectation_distance_aware:
            err = (exp_score - labels.float()).abs().detach()
            p = float(expectation_distance_power)
            if p <= 0.0:
                dist_w = 1.0 + err
            else:
                dist_w = 1.0 + err.pow(p)
            exp_loss = exp_loss * dist_w
        total = total + expectation_weight * exp_loss.mean()
    return total
