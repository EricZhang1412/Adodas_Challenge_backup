from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mtcn_backbone import MTCNBackbone, BackboneConfig


class ParticipantAggregator(nn.Module):

    def __init__(self, d_in: int, d_out: int, method: str = "mlp", dropout: float = 0.2):
        super().__init__()
        self.method = method
        self.d_in = d_in
        self.d_out = d_out

        if method == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(d_in, 4*d_out),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4*d_out, d_out),
            )
        elif method == "attention":
            self.query = nn.Linear(d_in, 1)
            self.proj = nn.Linear(d_in, d_out)
        elif method == "mean":
            self.proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        elif method == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_in,
                nhead=4,
                dim_feedforward=2 * d_in,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.sa = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def forward(self, session_reprs: torch.Tensor, session_valid: torch.Tensor) -> torch.Tensor:
        mask = session_valid.float().unsqueeze(-1)  
        masked_reprs = session_reprs * mask

        if self.method == "mean":
            n_valid = mask.sum(dim=1).clamp(min=1)  
            pooled = masked_reprs.sum(dim=1) / n_valid  
            return self.proj(pooled)

        elif self.method == "mlp":
            n_valid = mask.sum(dim=1).clamp(min=1)
            pooled = masked_reprs.sum(dim=1) / n_valid
            return self.mlp(pooled)

        elif self.method == "attention":
            scores = self.query(session_reprs).squeeze(-1)
            scores = scores.masked_fill(~session_valid, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            weights = weights.masked_fill(~session_valid, 0.0)
            pooled = (weights.unsqueeze(-1) * session_reprs).sum(dim=1)
            return self.proj(pooled)

        elif self.method == "transformer":
            key_padding_mask = ~session_valid  # (B, 4): True = ignore position
            out = self.sa(session_reprs, src_key_padding_mask=key_padding_mask)
            n_valid = mask.sum(dim=1).clamp(min=1)
            pooled = (out * mask).sum(dim=1) / n_valid
            return self.proj(pooled)


class SessionTypeClassifier(nn.Module):
    def __init__(self, d_in: int, n_classes: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.GELU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DASSubscaleHead(nn.Module):
    """Auxiliary regression head: predicts D/A/S subscale raw sums from participant repr.

    DASS-21 subscale assignments (0-indexed, D01=0 … D21=20):
      Depression (7): D03,D05,D10,D13,D16,D17,D21 → [2,4,9,12,15,16,20]
      Anxiety    (7): D02,D04,D07,D09,D15,D19,D20 → [1,3,6,8,14,18,19]
      Stress     (7): D01,D06,D08,D11,D12,D14,D18 → [0,5,7,10,11,13,17]
    Each subscale sum ∈ [0, 21] (7 items × max 3).
    """

    depression_idx: list[int] = [2, 4, 9, 12, 15, 16, 20]
    anxiety_idx: list[int]    = [1, 3, 6, 8, 14, 18, 19]
    stress_idx: list[int]     = [0, 5, 7, 10, 11, 13, 17]

    def __init__(self, d_in: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_in // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_in // 2, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, d_in) → (B, 3) predicted subscale sums."""
        return self.fc(x)

    @classmethod
    def compute_targets(cls, labels: torch.Tensor) -> torch.Tensor:
        """(B, 21) int labels → (B, 3) float subscale sum targets."""
        lf = labels.float()
        d = lf[:, cls.depression_idx].sum(dim=-1)
        a = lf[:, cls.anxiety_idx].sum(dim=-1)
        s = lf[:, cls.stress_idx].sum(dim=-1)
        return torch.stack([d, a, s], dim=-1)


class GroupedModel(nn.Module):

    def __init__(
        self,
        backbone: MTCNBackbone,
        d_shared: int,
        aggregator_method: str = "mlp",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = backbone
        self.aggregator = ParticipantAggregator(
            d_in=d_shared, d_out=d_shared,
            method=aggregator_method, dropout=dropout,
        )
        self.session_type_head = SessionTypeClassifier(d_in=d_shared)

    def forward(
        self,
        flat_batch: dict,
        n_participants: int,
        session_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        session_reprs = self.backbone(flat_batch) 

        B = n_participants
        session_grid = session_reprs.view(B, 4, -1)


        participant_repr = self.aggregator(session_grid, session_valid) 

        session_type_logits = self.session_type_head(session_reprs) 

        return {
            "session_reprs": session_reprs,
            "participant_repr": participant_repr,
            "session_type_logits": session_type_logits,
        }


class CORALHead(nn.Module):

    def __init__(
        self,
        d_in: int,
        n_items: int = 21,
        n_thresholds: int = 3,
        threshold_init: torch.Tensor | None = None,
    ):
        """CORAL ordinal head with one shared score per item and per-item monotonic thresholds.

        Args:
            threshold_init: optional (n_items, n_thresholds) tensor of *target*
                cumulative threshold values. If provided, raw_thresholds is
                seeded so that softplus → cumsum reproduces these targets at
                init. Typical use: pass logit(P(y < k)) computed from training
                frequencies so a score of 0 reproduces the marginal class
                distribution (much better than the constant 0.5 default).
        """
        super().__init__()
        self.n_items = n_items
        self.n_thresholds = n_thresholds

        self.score_fc = nn.Linear(d_in, n_items)

        self.raw_thresholds = nn.Parameter(torch.zeros(n_items, n_thresholds))
        if threshold_init is not None:
            init = torch.as_tensor(threshold_init, dtype=torch.float32)
            assert init.shape == (n_items, n_thresholds), (
                f"threshold_init shape {tuple(init.shape)} != "
                f"({n_items}, {n_thresholds})"
            )
            # Convert target cumulative thresholds → softplus^-1(spacings).
            # spacings_k = thresholds_k − thresholds_{k-1} (clamped > 0 to keep
            # softplus^-1 finite; the clamp is only relevant for items whose
            # P(y >= k) > 0.5 cases — rare for DASS-21).
            spacings = torch.zeros_like(init)
            spacings[..., 0] = init[..., 0]
            spacings[..., 1:] = init[..., 1:] - init[..., :-1]
            spacings = spacings.clamp(min=1e-3)
            # softplus^-1(y) = log(exp(y) − 1) = log(expm1(y))
            raw_init = torch.log(torch.expm1(spacings).clamp(min=1e-9))
            with torch.no_grad():
                self.raw_thresholds.copy_(raw_init)
        else:
            nn.init.constant_(self.raw_thresholds, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.score_fc(x)

        spacings = F.softplus(self.raw_thresholds) 
        thresholds = torch.cumsum(spacings, dim=-1) 

        logits = scores.unsqueeze(-1) - thresholds.unsqueeze(0) 
        return logits

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
