import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


PLAN_NAMES = ["exploit", "explore", "balance", "recover"]


def derive_plan_target(
    *,
    slate_credit: float,
    mean_support: float,
    mean_response: float,
    top_share: float,
) -> Tuple[int, float]:
    credit = float(slate_credit)
    support = float(mean_support)
    response = float(mean_response)
    share = float(top_share)
    if credit < -1e-6:
        label = 3  # recover
    elif support < 0.58:
        label = 1  # explore
    elif support >= 0.72 and share >= 0.38:
        label = 0  # exploit
    else:
        label = 2  # balance
    base_ratio = [0.22, 0.30, 0.18, 0.35][label]
    response_bonus = 0.05 * float(np.clip(response, 0.0, 1.0))
    if label == 3:
        response_bonus = 0.0
    ratio = float(np.clip(base_ratio + response_bonus, 0.10, 0.45))
    return int(label), ratio


class HistoryPlanHead(nn.Module):
    def __init__(self, hidden_size: int, n_plans: int, *, mlp_dim: int = 128):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.n_plans = int(n_plans)
        self.mlp_dim = int(mlp_dim)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, self.mlp_dim)
        self.act = nn.Tanh()
        self.logit_head = nn.Linear(self.mlp_dim, self.n_plans)
        self.value_head = nn.Linear(self.mlp_dim, 1)

    def forward(self, history_summary: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.norm(history_summary)
        x = self.fc1(x)
        x = self.act(x)
        logits = self.logit_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value


class SlateCreditHead(nn.Module):
    def __init__(self, hidden_size: int, *, mlp_dim: int = 128):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.mlp_dim = int(mlp_dim)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, self.mlp_dim)
        self.act = nn.Tanh()
        self.value_head = nn.Linear(self.mlp_dim, 1)

    def forward(self, history_summary: torch.Tensor) -> torch.Tensor:
        x = self.norm(history_summary)
        x = self.fc1(x)
        x = self.act(x)
        return self.value_head(x).squeeze(-1)


class _PlanConditionedTokenBase(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        n_plans: int,
        *,
        token_dim: int = 32,
        plan_dim: int = 16,
        mlp_dim: int = 128,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.n_plans = int(n_plans)
        self.token_dim = int(token_dim)
        self.plan_dim = int(plan_dim)
        self.mlp_dim = int(mlp_dim)
        self.token_emb = nn.Embedding(self.vocab_size, self.token_dim)
        self.plan_proj = nn.Linear(self.n_plans, self.plan_dim)
        self.norm = nn.LayerNorm(self.hidden_size + self.token_dim + self.plan_dim)
        self.fc1 = nn.Linear(self.hidden_size + self.token_dim + self.plan_dim, self.mlp_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(self.mlp_dim, 1)

    def _plan_ctx(self, plan_probs: torch.Tensor) -> torch.Tensor:
        return self.plan_proj(plan_probs.float())

    def _merge(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        plan_probs: torch.Tensor,
    ) -> torch.Tensor:
        tok = self.token_emb(token_ids)
        plan = self._plan_ctx(plan_probs)
        x = torch.cat([hidden_states, tok, plan], dim=-1)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x.squeeze(-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        plan_probs: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.dim() == 3:
            length = int(hidden_states.shape[1])
            plan = plan_probs.unsqueeze(1).expand(-1, length, -1)
            out = self._merge(
                hidden_states.reshape(-1, hidden_states.shape[-1]),
                token_ids.reshape(-1),
                plan.reshape(-1, plan.shape[-1]),
            )
            return out.view(hidden_states.shape[0], hidden_states.shape[1])
        return self._merge(hidden_states, token_ids, plan_probs)

    def score_all_tokens(self, hidden_states: torch.Tensor, plan_probs: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must have shape [B, D]")
        bsz = int(hidden_states.shape[0])
        token_ids = torch.arange(self.vocab_size, device=hidden_states.device, dtype=torch.long)
        hidden = hidden_states.unsqueeze(1).expand(bsz, self.vocab_size, hidden_states.shape[-1])
        tok = self.token_emb(token_ids).unsqueeze(0).expand(bsz, self.vocab_size, self.token_dim)
        plan = self._plan_ctx(plan_probs).unsqueeze(1).expand(bsz, self.vocab_size, self.plan_dim)
        x = torch.cat([hidden, tok, plan], dim=-1)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x).squeeze(-1)
        return x


class PlanConditionedPrefixValueHead(_PlanConditionedTokenBase):
    pass


class PlanConditionedTokenActorHead(_PlanConditionedTokenBase):
    pass


def load_phase6_joint_bundle(head_path: str, meta_path: str, device: torch.device) -> Tuple[Dict[str, nn.Module], Dict[str, Any]]:
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    n_plans = int(meta.get("n_plans", len(PLAN_NAMES)))
    hidden_size = int(meta["hidden_size"])
    vocab_size = int(meta["vocab_size"])
    token_dim = int(meta.get("token_dim", 32))
    plan_dim = int(meta.get("plan_dim", 16))
    mlp_dim = int(meta.get("mlp_dim", 128))

    plan_head = HistoryPlanHead(hidden_size, n_plans, mlp_dim=int(meta.get("plan_mlp_dim", mlp_dim)))
    slate_head = SlateCreditHead(hidden_size, mlp_dim=int(meta.get("slate_mlp_dim", mlp_dim)))
    prefix_head = PlanConditionedPrefixValueHead(
        hidden_size,
        vocab_size,
        n_plans,
        token_dim=token_dim,
        plan_dim=plan_dim,
        mlp_dim=mlp_dim,
    )
    token_actor_head = PlanConditionedTokenActorHead(
        hidden_size,
        vocab_size,
        n_plans,
        token_dim=token_dim,
        plan_dim=plan_dim,
        mlp_dim=mlp_dim,
    )

    payload = torch.load(head_path, map_location=device)
    plan_head.load_state_dict(payload["plan_head_state_dict"])
    slate_head.load_state_dict(payload["slate_head_state_dict"])
    prefix_head.load_state_dict(payload["prefix_head_state_dict"])
    token_actor_head.load_state_dict(payload["token_actor_head_state_dict"])

    for module in (plan_head, slate_head, prefix_head, token_actor_head):
        module.to(device)
        module.eval()

    return {
        "plan_head": plan_head,
        "slate_head": slate_head,
        "prefix_head": prefix_head,
        "token_actor_head": token_actor_head,
    }, meta
