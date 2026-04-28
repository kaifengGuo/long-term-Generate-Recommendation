import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class TokenResidualActorHead(nn.Module):
    """Token-aware residual actor head for decode-time policy nudging."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        token_dim: int = 32,
        mlp_dim: int = 128,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.token_dim = int(token_dim)
        self.mlp_dim = int(mlp_dim)
        self.token_emb = nn.Embedding(self.vocab_size, self.token_dim)
        self.norm = nn.LayerNorm(self.hidden_size + self.token_dim)
        self.fc1 = nn.Linear(self.hidden_size + self.token_dim, self.mlp_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(self.mlp_dim, 1)

    def _merge(self, hidden_states: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        tok = self.token_emb(token_ids)
        x = torch.cat([hidden_states, tok], dim=-1)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x.squeeze(-1)

    def forward(self, hidden_states: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 3:
            out = self._merge(
                hidden_states.reshape(-1, hidden_states.shape[-1]),
                token_ids.reshape(-1),
            )
            return out.view(hidden_states.shape[0], hidden_states.shape[1])
        return self._merge(hidden_states, token_ids)

    def score_all_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must have shape [B, D]")
        bsz = int(hidden_states.shape[0])
        token_ids = torch.arange(self.vocab_size, device=hidden_states.device, dtype=torch.long)
        hidden = hidden_states.unsqueeze(1).expand(bsz, self.vocab_size, hidden_states.shape[-1])
        tok = self.token_emb(token_ids).unsqueeze(0).expand(bsz, self.vocab_size, self.token_dim)
        x = torch.cat([hidden, tok], dim=-1)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x).squeeze(-1)
        return x


def load_phase5_token_actor_head(head_path: str, meta_path: str, device: torch.device) -> Tuple[TokenResidualActorHead, Dict[str, Any]]:
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    head = TokenResidualActorHead(
        hidden_size=int(meta["hidden_size"]),
        vocab_size=int(meta["vocab_size"]),
        token_dim=int(meta.get("token_dim", 32)),
        mlp_dim=int(meta.get("mlp_dim", 128)),
    )
    payload = torch.load(head_path, map_location=device)
    state = payload.get("model_state_dict", payload)
    head.load_state_dict(state)
    head = head.to(device)
    head.eval()
    return head, meta
