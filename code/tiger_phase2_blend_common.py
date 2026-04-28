import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class TIGER(nn.Module):
    """T5 backbone over SID token sequences."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        t5cfg = T5Config(
            num_layers=int(config["num_layers"]),
            num_decoder_layers=int(config["num_decoder_layers"]),
            d_model=int(config["d_model"]),
            d_ff=int(config["d_ff"]),
            num_heads=int(config["num_heads"]),
            d_kv=int(config["d_kv"]),
            dropout_rate=float(config["dropout_rate"]),
            vocab_size=int(config["vocab_size"]),
            pad_token_id=int(config["pad_token_id"]),
            eos_token_id=int(config["eos_token_id"]),
            decoder_start_token_id=int(config["pad_token_id"]),
            feed_forward_proj=str(config.get("feed_forward_proj", "relu")),
        )
        self.model = T5ForConditionalGeneration(t5cfg)

    @property
    def n_parameters(self) -> str:
        num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
        total_params = num_params(self.parameters())
        emb_params = num_params(self.model.get_input_embeddings().parameters())
        return (
            f"#Embedding parameters: {emb_params}\n"
            f"#Non-embedding parameters: {total_params - emb_params}\n"
            f"#Total trainable parameters: {total_params}\n"
        )

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        out = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return out.last_hidden_state

    def decode_with_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        decoder_input_ids: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        return out.logits, out.decoder_hidden_states[-1]

    def decode_with_hidden_from_encoded(
        self,
        encoder_hidden: torch.Tensor,
        attention_mask: torch.Tensor = None,
        decoder_input_ids: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.model(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden),
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        return out.logits, out.decoder_hidden_states[-1]

    def decode_step_with_cache_from_encoded(
        self,
        encoder_hidden: torch.Tensor,
        attention_mask: torch.Tensor = None,
        decoder_input_ids: torch.Tensor = None,
        past_key_values: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        if past_key_values is not None and isinstance(past_key_values, tuple):
            try:
                from transformers.cache_utils import EncoderDecoderCache

                past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)
            except Exception:
                pass
        out = self.model(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden),
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
        )
        return out.logits, out.decoder_hidden_states[-1], out.past_key_values

    def reorder_cache(self, past_key_values: Any, beam_idx: torch.Tensor) -> Any:
        if past_key_values is None:
            return None
        if hasattr(past_key_values, "to_legacy_cache"):
            legacy_cache = past_key_values.to_legacy_cache()
            reordered = self._reorder_legacy_cache(legacy_cache, beam_idx)
            try:
                from transformers.cache_utils import EncoderDecoderCache

                return EncoderDecoderCache.from_legacy_cache(reordered)
            except Exception:
                return reordered
        return self._reorder_legacy_cache(past_key_values, beam_idx)

    @staticmethod
    def _reorder_legacy_cache(past_key_values: Any, beam_idx: torch.Tensor) -> Any:
        reordered_layers = []
        for layer_past in past_key_values:
            reordered_states = []
            for state in layer_past:
                if state is None:
                    reordered_states.append(None)
                else:
                    reordered_states.append(state.index_select(0, beam_idx))
            reordered_layers.append(tuple(reordered_states))
        return tuple(reordered_layers)


class TokenCreditTransportHead(nn.Module):
    """Predict signed SID-block credit for each target token under decode context."""

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


class TokenLongTermActorHead(nn.Module):
    """Predict decode-time token logits for a lightweight long-term actor."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        mlp_dim: int = 256,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.mlp_dim = int(mlp_dim)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, self.mlp_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(self.mlp_dim, self.vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.norm(hidden_states)
        x = self.fc1(x)
        x = self.act(x)
        return self.fc2(x)


class TokenPrefixValueHead(nn.Module):
    """Predict cumulative prefix value for each generated SID token."""

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


def load_tiger_model(
    *,
    tiger_ckpt: str,
    sid_mapping_path: str,
    num_layers: int,
    num_decoder_layers: int,
    d_model: int,
    d_ff: int,
    num_heads: int,
    d_kv: int,
    dropout_rate: float,
    feed_forward_proj: str,
    device: torch.device,
) -> Tuple[TIGER, int, int]:
    payload = torch.load(tiger_ckpt, map_location=device)
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    df_sid = pd.read_csv(sid_mapping_path)
    sid_cols = [c for c in df_sid.columns if c.startswith("sid_")]
    if not sid_cols:
        raise ValueError(f"No sid_* columns found in {sid_mapping_path}")
    sid_depth = len(sid_cols)
    codes_raw = df_sid[sid_cols].values.astype(int)
    codebook_size = int(codes_raw.max() + 1)
    vocab_size = codebook_size + 1
    payload_cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
    config = {
        "num_layers": int(payload_cfg.get("num_layers", num_layers)),
        "num_decoder_layers": int(payload_cfg.get("num_decoder_layers", num_decoder_layers)),
        "d_model": int(payload_cfg.get("d_model", d_model)),
        "d_ff": int(payload_cfg.get("d_ff", d_ff)),
        "num_heads": int(payload_cfg.get("num_heads", num_heads)),
        "d_kv": int(payload_cfg.get("d_kv", d_kv)),
        "dropout_rate": float(payload_cfg.get("dropout_rate", dropout_rate)),
        "feed_forward_proj": str(payload_cfg.get("feed_forward_proj", feed_forward_proj)),
        "vocab_size": int(payload_cfg.get("vocab_size", vocab_size)),
        "pad_token_id": int(payload_cfg.get("pad_token_id", 0)),
        "eos_token_id": int(payload_cfg.get("eos_token_id", 0)),
    }
    model = TIGER(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, int(sid_depth), int(codebook_size)


def build_iid2sid_tokens(reader, mapping_path: str, sid_depth: int, device: torch.device) -> Tuple[torch.Tensor, Dict[Tuple[int, ...], int]]:
    df = pd.read_csv(mapping_path)
    sid_cols = [f"sid_{i + 1}" for i in range(sid_depth)]
    if not all(c in df.columns for c in sid_cols):
        sid_cols = [f"sid_{i}" for i in range(sid_depth)]
    if not all(c in df.columns for c in sid_cols):
        raise ValueError(f"SID columns not found in mapping: {sid_cols}")

    df = df[["video_id"] + sid_cols].drop_duplicates("video_id").set_index("video_id")
    n_items = len(reader.items)
    iid2sid_raw = np.zeros((n_items + 1, sid_depth), dtype=np.int64)
    sid2iid_map_tok: Dict[Tuple[int, ...], int] = {}

    for vid, iid in reader.item_id_vocab.items():
        if vid in df.index:
            vals = df.loc[vid, sid_cols].astype(int).values
            iid2sid_raw[int(iid)] = vals

    iid2sid_tok = np.zeros_like(iid2sid_raw, dtype=np.int64)
    iid2sid_tok[1:] = iid2sid_raw[1:] + 1

    for iid in range(1, n_items + 1):
        codes_tok = tuple(int(x) for x in iid2sid_tok[iid].tolist())
        if any(c > 0 for c in codes_tok):
            sid2iid_map_tok[codes_tok] = int(iid)
    return torch.from_numpy(iid2sid_tok).to(device), sid2iid_map_tok


def build_sid_prefix_to_next(sid2iid_map_tok: Dict[Tuple[int, ...], int]) -> Dict[Tuple[int, ...], List[int]]:
    trie: Dict[Tuple[int, ...], List[int]] = {}
    for sid_seq in sid2iid_map_tok.keys():
        for pos in range(len(sid_seq)):
            prefix = tuple(sid_seq[:pos])
            nxt = int(sid_seq[pos])
            bucket = trie.setdefault(prefix, [])
            if nxt not in bucket:
                bucket.append(nxt)
    for prefix in trie:
        trie[prefix] = sorted(trie[prefix])
    return trie


def build_history_tokens(hist_iids: torch.Tensor, iid2sid_tok: torch.Tensor, max_hist_items: int, sid_depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, hist_env = hist_iids.shape
    device = hist_iids.device
    hist_iids = hist_iids.clamp(min=0, max=iid2sid_tok.size(0) - 1)
    keep = min(int(hist_env), int(max_hist_items))
    if hist_env > keep:
        hist_inp = hist_iids[:, -keep:]
    else:
        pad_items = torch.zeros(bsz, int(max_hist_items) - hist_env, dtype=torch.long, device=device)
        hist_inp = torch.cat([pad_items, hist_iids], dim=1)
        keep = int(max_hist_items)
    hist_sids = iid2sid_tok[hist_inp]
    bsz, keep, depth = hist_sids.shape
    if int(depth) != int(sid_depth):
        raise ValueError(f"sid_depth mismatch: expected {sid_depth}, got {depth}")
    history = hist_sids.reshape(bsz, keep * depth)
    attention_mask = (history != 0).long()
    return history, attention_mask


def decoder_input_ids_from_targets(target_tokens: torch.Tensor) -> torch.Tensor:
    start = torch.zeros(target_tokens.size(0), 1, dtype=torch.long, device=target_tokens.device)
    if target_tokens.size(1) <= 1:
        return start
    return torch.cat([start, target_tokens[:, :-1]], dim=1)


def sinkhorn_transport(
    row_mass: np.ndarray,
    col_mass: np.ndarray,
    cost: np.ndarray,
    *,
    epsilon: float = 0.35,
    n_iter: int = 16,
) -> np.ndarray:
    if cost.size == 0:
        return np.zeros_like(cost, dtype=np.float32)
    r = np.asarray(row_mass, dtype=np.float64).reshape(-1)
    c = np.asarray(col_mass, dtype=np.float64).reshape(-1)
    r = np.clip(r, 1e-8, None)
    c = np.clip(c, 1e-8, None)
    r = r / max(float(r.sum()), 1e-8)
    c = c / max(float(c.sum()), 1e-8)
    kernel = np.exp(-np.asarray(cost, dtype=np.float64) / max(float(epsilon), 1e-6))
    kernel = np.clip(kernel, 1e-8, None)
    u = np.ones_like(r)
    v = np.ones_like(c)
    for _ in range(max(int(n_iter), 1)):
        u = r / np.maximum(kernel @ v, 1e-8)
        v = c / np.maximum(kernel.T @ u, 1e-8)
    plan = (u[:, None] * kernel) * v[None, :]
    total = float(plan.sum())
    if total > 0:
        plan = plan / total
    return plan.astype(np.float32)


def infer_model_size_args(model_size: str) -> Dict[str, Any]:
    name = str(model_size).lower()
    if name == "mini":
        return {"num_layers": 3, "num_decoder_layers": 3, "d_model": 128, "d_ff": 512, "num_heads": 4, "d_kv": 16}
    if name == "medium":
        return {"num_layers": 4, "num_decoder_layers": 4, "d_model": 128, "d_ff": 1024, "num_heads": 6, "d_kv": 64}
    if name == "large":
        return {"num_layers": 6, "num_decoder_layers": 6, "d_model": 192, "d_ff": 1536, "num_heads": 8, "d_kv": 24}
    raise ValueError(f"Unsupported model_size: {model_size}")


def load_phase2_head(head_path: str, meta_path: str, device: torch.device) -> Tuple[TokenCreditTransportHead, Dict[str, Any]]:
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    head = TokenCreditTransportHead(
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


def load_phase3_actor_head(head_path: str, meta_path: str, device: torch.device) -> Tuple[TokenLongTermActorHead, Dict[str, Any]]:
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    head = TokenLongTermActorHead(
        hidden_size=int(meta["hidden_size"]),
        vocab_size=int(meta["vocab_size"]),
        mlp_dim=int(meta.get("mlp_dim", 256)),
    )
    payload = torch.load(head_path, map_location=device)
    state = payload.get("model_state_dict", payload)
    head.load_state_dict(state)
    head = head.to(device)
    head.eval()
    return head, meta


def load_prefix_value_head(head_path: str, meta_path: str, device: torch.device) -> Tuple[TokenPrefixValueHead, Dict[str, Any]]:
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    head = TokenPrefixValueHead(
        hidden_size=int(meta["d_model"]),
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


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
