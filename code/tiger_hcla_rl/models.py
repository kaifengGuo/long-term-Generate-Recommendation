import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class PageLongTermCritic(nn.Module):
    def __init__(self, hidden_size: int, page_dim: int, *, mlp_dim: int = 128):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.page_dim = int(page_dim)
        self.mlp_dim = int(mlp_dim)
        self.hist_norm = nn.LayerNorm(self.hidden_size)
        self.page_norm = nn.LayerNorm(self.page_dim)
        self.fc1 = nn.Linear(self.hidden_size + self.page_dim, self.mlp_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.out = nn.Linear(self.mlp_dim, 1)

    def forward(self, history_summary: torch.Tensor, page_features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.hist_norm(history_summary), self.page_norm(page_features)], dim=-1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        return self.out(x).squeeze(-1)


class ItemLongTermCritic(nn.Module):
    def __init__(self, hidden_size: int, page_dim: int, item_dim: int, *, mlp_dim: int = 128):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.page_dim = int(page_dim)
        self.item_dim = int(item_dim)
        self.mlp_dim = int(mlp_dim)
        self.hist_norm = nn.LayerNorm(self.hidden_size)
        self.page_norm = nn.LayerNorm(self.page_dim)
        self.item_norm = nn.LayerNorm(self.item_dim)
        self.fc1 = nn.Linear(self.hidden_size + self.page_dim + self.item_dim, self.mlp_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.out = nn.Linear(self.mlp_dim, 1)

    def forward(
        self,
        history_summary: torch.Tensor,
        page_features: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                self.hist_norm(history_summary),
                self.page_norm(page_features),
                self.item_norm(item_features),
            ],
            dim=-1,
        )
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        return self.out(x).squeeze(-1)


class TokenLongTermCritic(nn.Module):
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
            out = self._merge(hidden_states.reshape(-1, hidden_states.shape[-1]), token_ids.reshape(-1))
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


def save_critic_bundle(
    bundle_path: str | Path,
    meta_path: str | Path,
    *,
    page_head: PageLongTermCritic,
    item_head: ItemLongTermCritic,
    token_head: TokenLongTermCritic,
    meta: Dict[str, Any],
) -> None:
    bundle_path = Path(bundle_path)
    meta_path = Path(meta_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "page_head": page_head.state_dict(),
            "item_head": item_head.state_dict(),
            "token_head": token_head.state_dict(),
        },
        bundle_path,
    )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_critic_bundle(
    bundle_path: str | Path,
    meta_path: str | Path,
    device: torch.device,
) -> Tuple[PageLongTermCritic, ItemLongTermCritic, TokenLongTermCritic, Dict[str, Any]]:
    payload = torch.load(bundle_path, map_location=device)
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    page_head = PageLongTermCritic(
        hidden_size=int(meta["hidden_size"]),
        page_dim=int(meta["page_dim"]),
        mlp_dim=int(meta.get("mlp_dim", 128)),
    )
    item_head = ItemLongTermCritic(
        hidden_size=int(meta["hidden_size"]),
        page_dim=int(meta["page_dim"]),
        item_dim=int(meta["item_dim"]),
        mlp_dim=int(meta.get("mlp_dim", 128)),
    )
    token_head = TokenLongTermCritic(
        hidden_size=int(meta["hidden_size"]),
        vocab_size=int(meta["vocab_size"]),
        token_dim=int(meta.get("token_dim", 32)),
        mlp_dim=int(meta.get("mlp_dim", 128)),
    )
    page_head.load_state_dict(payload["page_head"])
    item_head.load_state_dict(payload["item_head"])
    token_head.load_state_dict(payload["token_head"])
    page_head = page_head.to(device).eval()
    item_head = item_head.to(device).eval()
    token_head = token_head.to(device).eval()
    return page_head, item_head, token_head, meta
