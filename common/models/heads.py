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
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """Per-threshold ordinal BCE with optional focal modulation.

    focal_gamma > 0 down-weights easy examples (well-predicted thresholds) so
    the gradient concentrates on hard ones — useful for the high-severity
    tails (k=2) where v5 sees per-item QWK ceilings around 0.32 even though
    pos_weight is already large. Default 0.0 == standard BCE.
    """
    targets = A2OrdinalHead.build_ordinal_targets(labels, n_thresholds=logits.size(-1))
    if label_smoothing > 0.0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
    if focal_gamma <= 0.0:
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

    # Focal: (1 - p_t)^gamma * BCE, where p_t is the prob assigned to the true
    # label. We compute BCE per element so the modulator stays per-element.
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    p = torch.sigmoid(logits)
    p_t = targets * p + (1.0 - targets) * (1.0 - p)
    modulator = (1.0 - p_t).clamp(min=1e-6).pow(focal_gamma)
    return (modulator * bce).mean()


def cumulative_to_class_probs(cum_logits: torch.Tensor) -> torch.Tensor:
    """Convert cumulative ordinal logits P(y>=k) for k=1..K-1 to K-class probabilities.

    Enforces monotonicity via element-wise min on the survival function so that
    the resulting class probabilities are non-negative and sum to 1.

    Args:
        cum_logits: (..., K-1) raw logits whose sigmoid models P(y >= k).

    Returns:
        (..., K) probability tensor.
    """
    s = torch.sigmoid(cum_logits)
    # Enforce monotonic survival function: s_k <= s_{k-1}
    s_mono = [s[..., 0]]
    for k in range(1, s.size(-1)):
        s_mono.append(torch.min(s[..., k], s_mono[-1]))
    s_mono = torch.stack(s_mono, dim=-1)  # (..., K-1)

    # P(y=0) = 1 - s_1; P(y=k) = s_k - s_{k+1}; P(y=K-1) = s_{K-1}
    p0 = (1.0 - s_mono[..., 0]).unsqueeze(-1)
    p_mid = s_mono[..., :-1] - s_mono[..., 1:]
    p_last = s_mono[..., -1:]
    probs = torch.cat([p0, p_mid, p_last], dim=-1)
    return probs.clamp(min=0.0)


def build_soft_labels(
    labels: torch.Tensor,
    n_classes: int = 4,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Distance-aware soft labels: w_c = exp(-((c - y)/sigma)^2 / 2), softmax-normalized.

    Args:
        labels: integer labels in [0, n_classes-1], any leading shape.
        n_classes: number of ordinal classes (4 for DASS-21).
        sigma: kernel width. Smaller -> sharper, closer to one-hot.

    Returns:
        Soft probability targets with shape labels.shape + (n_classes,).
    """
    classes = torch.arange(n_classes, device=labels.device, dtype=torch.float32)
    diff = labels.unsqueeze(-1).float() - classes
    log_w = -0.5 * (diff / max(sigma, 1e-6)) ** 2
    return F.softmax(log_w, dim=-1)


def soft_label_ce_loss(
    cum_logits: torch.Tensor,
    labels: torch.Tensor,
    sigma: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cross-entropy between Gaussian-soft targets and class probs derived from cumulative logits.

    Distance-aware: penalizes far-off predictions more than near ones, so it aligns better
    with the QWK quadratic weight matrix than per-threshold BCE.
    """
    n_classes = cum_logits.size(-1) + 1
    class_probs = cumulative_to_class_probs(cum_logits)
    soft_targets = build_soft_labels(labels, n_classes=n_classes, sigma=sigma)
    log_probs = torch.log(class_probs.clamp(min=eps))
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def emd_ordinal_loss(
    cum_logits: torch.Tensor,
    labels: torch.Tensor,
    p: int = 2,
) -> torch.Tensor:
    """Earth Mover's Distance loss on the ordinal cumulative form.

    For one-hot true label y with K classes, the survival function S_true(k) = 1[y >= k]
    is exactly build_ordinal_targets. With S_pred(k) = sigmoid(logit_k), the squared L2
    distance between predicted and true CDFs simplifies to MSE between S_pred and S_true:

        ||CDF_pred - CDF_true||_p^p = sum_k |S_pred(k) - S_true(k)|^p   (k=1..K-1)

    This is naturally distance-aware and matches CORAL structure without going through
    the 4-class projection.
    """
    targets = A2OrdinalHead.build_ordinal_targets(labels, n_thresholds=cum_logits.size(-1))
    s_pred = torch.sigmoid(cum_logits)
    diff = s_pred - targets
    if p == 1:
        loss = diff.abs().sum(dim=-1)
    else:
        loss = (diff ** 2).sum(dim=-1)
    return loss.mean()


def a2_combined_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    bce_weight: float = 1.0,
    soft_ce_weight: float = 0.0,
    emd_weight: float = 0.0,
    soft_label_sigma: float = 1.0,
    emd_norm: int = 2,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """A2 loss combining ordinal BCE (optionally focal) + Gaussian soft-label CE + EMD.

    Setting soft_ce_weight=0, emd_weight=0, focal_gamma=0 reproduces
    a2_ordinal_loss exactly.
    """
    total: torch.Tensor | float = 0.0

    if bce_weight > 0.0:
        total = total + bce_weight * a2_ordinal_loss(
            logits, labels,
            pos_weight=pos_weight,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
        )

    if soft_ce_weight > 0.0:
        total = total + soft_ce_weight * soft_label_ce_loss(
            logits, labels, sigma=soft_label_sigma
        )

    if emd_weight > 0.0:
        total = total + emd_weight * emd_ordinal_loss(logits, labels, p=emd_norm)

    if isinstance(total, float):
        # All weights zero — return a zero tensor on the right device/dtype.
        return logits.new_zeros(())
    return total
