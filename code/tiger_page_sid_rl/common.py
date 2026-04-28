import json
import random
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import utils  # noqa: E402
from reader import *  # noqa: F401,F403,E402

from tiger_phase2_blend_common import build_history_tokens  # noqa: E402
from tiger_slate_allocator_common import build_bootstrap_target_shares, compute_item_support_features, prepare_history_sid  # noqa: E402


JSONL_MANIFEST_TYPE = "jsonl_manifest"


def load_reader_from_uirm_log(uirm_log_path: str, device: str):
    with open(uirm_log_path, "r", encoding="utf-8") as infile:
        class_args = eval(infile.readline(), {"Namespace": Namespace})
        training_args = eval(infile.readline(), {"Namespace": Namespace})
    training_args.val_holdout_per_user = 0
    training_args.test_holdout_per_user = 0
    training_args.device = device
    reader_class = eval("{0}.{0}".format(class_args.reader))
    return reader_class(training_args)


def set_random_seed(seed: int) -> None:
    utils.set_random_seed(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_jsonl_sources(path: str | Path) -> List[Path]:
    src_path = Path(path)
    if src_path.suffix == ".json" and src_path.exists():
        try:
            payload = json.loads(src_path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, dict) and str(payload.get("type", "")).strip().lower() == JSONL_MANIFEST_TYPE:
            raw_paths = payload.get("paths", [])
            sources: List[Path] = []
            for raw in raw_paths:
                child = Path(str(raw))
                if not child.is_absolute():
                    child = (src_path.parent / child).resolve()
                sources.append(child)
            return sources
    return [src_path]


def write_jsonl_manifest(path: str | Path, src_paths: Sequence[str | Path]) -> Dict[str, Any]:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_paths = [str(Path(p).resolve()) for p in src_paths]
    payload = {
        "type": JSONL_MANIFEST_TYPE,
        "paths": resolved_paths,
        "num_paths": int(len(resolved_paths)),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def iter_jsonl_records(path: str | Path, max_rows: int = 0) -> Iterator[Dict[str, Any]]:
    n_rows = 0
    for src in resolve_jsonl_sources(path):
        if not src.exists():
            continue
        with src.open("r", encoding="utf-8") as infile:
            for line in infile:
                if int(max_rows) > 0 and n_rows >= int(max_rows):
                    return
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
                n_rows += 1


def load_jsonl_rows(path: str | Path, max_rows: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for payload in iter_jsonl_records(path, max_rows=int(max_rows)):
        rows.append(payload)
    return rows


def append_jsonl(dst_path: str | Path, src_path: str | Path) -> int:
    dst = Path(dst_path)
    src = Path(src_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    n_lines = 0
    with src.open("r", encoding="utf-8") as src_fp, dst.open("a", encoding="utf-8") as dst_fp:
        for line in src_fp:
            line = line.strip()
            if not line:
                continue
            dst_fp.write(line + "\n")
            n_lines += 1
    return int(n_lines)


def split_groups(groups: Sequence[str], valid_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    uniq = sorted(set(str(x) for x in groups))
    rng = random.Random(int(seed))
    rng.shuffle(uniq)
    n_valid = max(1, int(round(len(uniq) * float(valid_ratio)))) if len(uniq) > 1 else 0
    valid_groups = set(uniq[:n_valid])
    train_idx, valid_idx = [], []
    for idx, g in enumerate(groups):
        (valid_idx if str(g) in valid_groups else train_idx).append(idx)
    if not train_idx:
        train_idx, valid_idx = valid_idx[1:], valid_idx[:1]
    if not valid_idx:
        valid_idx = train_idx[:1]
    return np.asarray(train_idx, dtype=np.int64), np.asarray(valid_idx, dtype=np.int64)


def discounted_returns(step_rewards: Sequence[float], gamma: float) -> List[float]:
    returns = [0.0 for _ in step_rewards]
    running = 0.0
    for idx in range(len(step_rewards) - 1, -1, -1):
        running = float(step_rewards[idx]) + float(gamma) * running
        returns[idx] = float(running)
    return returns


def normalize_probabilities(
    values: Sequence[float] | np.ndarray,
    *,
    mask: Sequence[bool] | np.ndarray | None = None,
    use_abs: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if use_abs:
        arr = np.abs(arr)
    else:
        arr = np.maximum(arr, 0.0)
    if mask is None:
        mask_arr = np.ones_like(arr, dtype=bool)
    else:
        mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
        if mask_arr.shape[0] != arr.shape[0]:
            mask_arr = np.ones_like(arr, dtype=bool)
    arr = np.where(mask_arr, arr, 0.0)
    total = float(arr.sum())
    if total <= float(eps):
        arr = np.where(mask_arr, 1.0, 0.0).astype(np.float32)
        total = float(arr.sum())
    if total <= float(eps):
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / total).astype(np.float32)


def build_history_state(
    history_items: Sequence[int],
    *,
    iid2sid_tok_cpu: torch.Tensor,
    max_hist_items: int,
    sid_depth: int,
) -> Tuple[List[int], List[int]]:
    hist_tensor = torch.tensor([int(x) for x in history_items][-int(max_hist_items):], dtype=torch.long).view(1, -1)
    input_ids, attention_mask = build_history_tokens(
        hist_tensor,
        iid2sid_tok_cpu,
        int(max_hist_items),
        int(sid_depth),
    )
    return input_ids.squeeze(0).tolist(), attention_mask.squeeze(0).tolist()


def pooled_history_summary(tiger, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    enc_out = tiger.model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    hidden = enc_out.last_hidden_state
    mask = attention_mask.unsqueeze(-1).float()
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (hidden * mask).sum(dim=1) / denom


def align_selected_tokens(
    selected_item_ids: Sequence[int],
    selected_sid_tokens_list: Sequence[Sequence[int]],
    iid2sid_tok_cpu: torch.Tensor,
    sid_depth: int,
) -> List[List[int]]:
    out: List[List[int]] = []
    for item_idx, item_id in enumerate(selected_item_ids):
        if item_idx < len(selected_sid_tokens_list):
            tokens = [int(x) for x in selected_sid_tokens_list[item_idx]]
        elif 0 <= int(item_id) < int(iid2sid_tok_cpu.shape[0]):
            tokens = [int(x) for x in iid2sid_tok_cpu[int(item_id)].tolist()]
        else:
            tokens = []
        if len(tokens) != int(sid_depth):
            tokens = [0 for _ in range(int(sid_depth))]
        out.append(tokens)
    return out


def build_page_scalar_features(
    *,
    page_index: int,
    max_page_index: int,
    history_len: int,
    max_hist_items: int,
    slate_size: int,
    max_slate_size: int,
) -> np.ndarray:
    return np.asarray(
        [
            float(min(int(page_index), int(max_page_index)) / max(int(max_page_index), 1)),
            float(min(int(history_len), int(max_hist_items)) / max(int(max_hist_items), 1)),
            float(min(int(slate_size), int(max_slate_size)) / max(int(max_slate_size), 1)),
        ],
        dtype=np.float32,
    )


def get_user_feature_layout(reader) -> Tuple[List[str], List[int]]:
    if reader is None or not hasattr(reader, "selected_user_features"):
        return [], []
    user_vocab = getattr(reader, "user_vocab", {})
    keys = [f"uf_{feature_name}" for feature_name in list(reader.selected_user_features)]
    sizes: List[int] = []
    for key in keys:
        vocab_key = key[3:] if key.startswith("uf_") else key
        vocab = user_vocab.get(vocab_key, {})
        if vocab:
            first_value = next(iter(vocab.values()))
            sizes.append(int(np.asarray(first_value).reshape(-1).shape[0]))
        else:
            sizes.append(0)
    return keys, sizes


def build_user_feature_vector(
    reader,
    user_id: int,
    *,
    user_feature_keys: Sequence[str] | None = None,
    user_feature_sizes: Sequence[int] | None = None,
) -> np.ndarray:
    keys, sizes = list(user_feature_keys or []), list(user_feature_sizes or [])
    if not keys:
        keys, sizes = get_user_feature_layout(reader)
    if not keys:
        return np.zeros((0,), dtype=np.float32)

    total_dim = int(sum(int(x) for x in sizes))
    zero_vec = np.zeros((total_dim,), dtype=np.float32)
    if reader is None:
        return zero_vec

    try:
        user_meta = reader.get_user_meta_data(int(user_id))
    except Exception:
        return zero_vec

    flat_parts: List[np.ndarray] = []
    for key, expected_dim in zip(keys, sizes):
        value = user_meta.get(key, None)
        if value is None:
            flat_parts.append(np.zeros((int(expected_dim),), dtype=np.float32))
            continue
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if int(expected_dim) > 0 and int(arr.shape[0]) != int(expected_dim):
            if int(arr.shape[0]) > int(expected_dim):
                arr = arr[: int(expected_dim)]
            else:
                arr = np.pad(arr, (0, int(expected_dim) - int(arr.shape[0])))
        flat_parts.append(arr.astype(np.float32, copy=False))
    if not flat_parts:
        return zero_vec
    return np.concatenate(flat_parts, axis=0).astype(np.float32, copy=False)


def build_page_samples(
    *,
    trace_rows: Sequence[Dict[str, Any]],
    iid2sid_tok_cpu: torch.Tensor,
    reader=None,
    max_hist_items: int,
    gamma: float,
    hazard_lambda: float,
    max_episodes: int = 0,
    aux_target_heuristic_mix: float = 0.60,
    aux_target_support_mix: float = 0.25,
    aux_target_response_mix: float = 0.15,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sid_depth = int(iid2sid_tok_cpu.shape[1])
    trace_grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in trace_rows:
        if "episode_id" not in rec or "selected_item_ids" not in rec:
            continue
        trace_grouped.setdefault(str(rec["episode_id"]), []).append(rec)
    episode_ids = sorted(trace_grouped.keys(), key=lambda x: int(x))
    if int(max_episodes) > 0:
        episode_ids = episode_ids[: int(max_episodes)]

    max_page_index = 1
    max_slate_size = 1
    for episode_id in episode_ids:
        for rec in trace_grouped.get(episode_id, []):
            max_page_index = max(max_page_index, int(rec.get("page_index", 1)))
            max_slate_size = max(max_slate_size, len(rec.get("selected_item_ids", [])))

    user_feature_keys, user_feature_sizes = get_user_feature_layout(reader)
    user_feat_dim = int(sum(int(x) for x in user_feature_sizes))
    user_feature_cache: Dict[int, List[float]] = {}

    samples: List[Dict[str, Any]] = []
    for episode_id in episode_ids:
        pages = sorted(trace_grouped.get(episode_id, []), key=lambda x: int(x.get("page_index", 0)))
        if not pages:
            continue
        lt_rewards = [
            float(rec.get("step_reward", 0.0)) - float(hazard_lambda) * float(1.0 if bool(rec.get("done", False)) else 0.0)
            for rec in pages
        ]
        returns = discounted_returns(lt_rewards, float(gamma))
        for page_pos, rec in enumerate(pages):
            selected_item_ids = [int(x) for x in rec.get("selected_item_ids", [])]
            slate_size = len(selected_item_ids)
            if slate_size <= 0:
                continue
            token_ids = align_selected_tokens(
                selected_item_ids,
                rec.get("selected_sid_tokens_list", []),
                iid2sid_tok_cpu,
                int(sid_depth),
            )
            history_items = [int(x) for x in rec.get("history_items", [])][-int(max_hist_items):]
            pre_input_ids, pre_attention_mask = build_history_state(
                history_items,
                iid2sid_tok_cpu=iid2sid_tok_cpu,
                max_hist_items=int(max_hist_items),
                sid_depth=int(sid_depth),
            )
            selected_responses = [[float(v) for v in row] for row in rec.get("selected_responses", [])]
            if len(selected_responses) != slate_size:
                width = len(selected_responses[0]) if selected_responses else len(rec.get("response_weights", []))
                selected_responses = [[0.0 for _ in range(width)] for _ in range(slate_size)]
            reward_weights = [float(x) for x in rec.get("response_weights", [])]
            item_rewards = np.asarray(rec.get("selected_item_rewards", []), dtype=np.float32).reshape(-1)
            if item_rewards.shape[0] != slate_size:
                item_rewards = np.zeros((slate_size,), dtype=np.float32)
            _hist_items_valid, history_sid = prepare_history_sid(
                history_items,
                iid2sid_tok_cpu=iid2sid_tok_cpu,
                max_hist_items=int(max_hist_items),
            )
            support_strengths: List[float] = []
            for tokens in token_ids:
                _support_feat, strength = compute_item_support_features(history_sid, tokens)
                support_strengths.append(float(strength))
            item_share_target, heuristic_share_target, centered_rewards, response_strength_arr = build_bootstrap_target_shares(
                page_credit=float(returns[page_pos]),
                item_rewards=item_rewards.tolist(),
                responses=selected_responses,
                support_strengths=support_strengths,
                response_weights=reward_weights if reward_weights else None,
                heuristic_mix=float(aux_target_heuristic_mix),
                support_mix=float(aux_target_support_mix),
                response_mix=float(aux_target_response_mix),
            )
            item_adv_target = (item_share_target.astype(np.float32) * float(returns[page_pos])).astype(np.float32)
            sid_share_target = np.zeros((slate_size, int(sid_depth)), dtype=np.float32)
            sid_adv_target = np.zeros((slate_size, int(sid_depth)), dtype=np.float32)
            for item_idx, item_tokens in enumerate(token_ids):
                token_mask = np.asarray([int(x) > 0 for x in item_tokens], dtype=bool)
                sid_share = normalize_probabilities(
                    np.ones((int(sid_depth),), dtype=np.float32),
                    mask=token_mask,
                    use_abs=False,
                )
                sid_share_target[item_idx] = sid_share
                sid_adv_target[item_idx] = sid_share * float(item_adv_target[item_idx])
            page_features = build_page_scalar_features(
                page_index=int(rec.get("page_index", page_pos + 1)),
                max_page_index=int(max_page_index),
                history_len=len(history_items),
                max_hist_items=int(max_hist_items),
                slate_size=int(slate_size),
                max_slate_size=int(max_slate_size),
            )
            user_id = int(rec.get("user_id", -1))
            if int(user_feat_dim) > 0:
                if int(user_id) not in user_feature_cache:
                    user_feature_cache[int(user_id)] = build_user_feature_vector(
                        reader,
                        int(user_id),
                        user_feature_keys=user_feature_keys,
                        user_feature_sizes=user_feature_sizes,
                    ).tolist()
                user_features = list(user_feature_cache[int(user_id)])
            else:
                user_features = []
            samples.append(
                {
                    "group": str(episode_id),
                    "episode_id": int(episode_id),
                    "user_id": int(user_id),
                    "page_index": int(rec.get("page_index", page_pos + 1)),
                    "history_items": list(history_items),
                    "pre_input_ids": list(pre_input_ids),
                    "pre_attention_mask": list(pre_attention_mask),
                    "selected_item_ids": list(selected_item_ids),
                    "token_ids": [list(tokens) for tokens in token_ids],
                    "page_features": page_features.tolist(),
                    "user_features": list(user_features),
                    "q_target": float(returns[page_pos]),
                    "item_rewards": item_rewards.astype(np.float32).tolist(),
                    "response_strengths": response_strength_arr.astype(np.float32).tolist(),
                    "support_strengths": [float(x) for x in support_strengths],
                    "item_share_target": item_share_target.astype(np.float32).tolist(),
                    "item_share_target_heuristic": heuristic_share_target.astype(np.float32).tolist(),
                    "item_centered_rewards": centered_rewards.astype(np.float32).tolist(),
                    "item_adv_target": item_adv_target.astype(np.float32).tolist(),
                    "sid_share_target": sid_share_target.astype(np.float32).tolist(),
                    "sid_adv_target": sid_adv_target.astype(np.float32).tolist(),
                    "lt_reward": float(lt_rewards[page_pos]),
                    "step_reward": float(rec.get("step_reward", 0.0)),
                    "done": bool(rec.get("done", False)),
                }
            )
    meta = {
        "n_trace_rows": int(len(trace_rows)),
        "n_pages": int(len(samples)),
        "sid_depth": int(sid_depth),
        "page_feat_dim": 3,
        "user_feat_dim": int(user_feat_dim),
        "user_feature_keys": list(user_feature_keys),
        "max_page_index": int(max_page_index),
        "max_slate_size": int(max_slate_size),
        "gamma": float(gamma),
        "hazard_lambda": float(hazard_lambda),
        "aux_target_heuristic_mix": float(aux_target_heuristic_mix),
        "aux_target_support_mix": float(aux_target_support_mix),
        "aux_target_response_mix": float(aux_target_response_mix),
    }
    return samples, meta


class TracePageDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[int(idx)]


def collate_trace_pages(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    sid_depth = len(batch[0]["token_ids"][0]) if batch and batch[0]["token_ids"] else 0
    max_items = max(len(row["token_ids"]) for row in batch)
    user_feat_dim = len(batch[0].get("user_features", [])) if batch else 0
    token_ids = torch.zeros((len(batch), max_items, sid_depth), dtype=torch.long)
    item_mask = torch.zeros((len(batch), max_items), dtype=torch.bool)
    item_share_target = torch.zeros((len(batch), max_items), dtype=torch.float32)
    item_adv_target = torch.zeros((len(batch), max_items), dtype=torch.float32)
    sid_share_target = torch.zeros((len(batch), max_items, sid_depth), dtype=torch.float32)
    sid_adv_target = torch.zeros((len(batch), max_items, sid_depth), dtype=torch.float32)
    for row_idx, row in enumerate(batch):
        for item_idx, item_tokens in enumerate(row["token_ids"]):
            item_mask[row_idx, item_idx] = any(int(x) > 0 for x in item_tokens)
            token_ids[row_idx, item_idx] = torch.tensor(item_tokens, dtype=torch.long)
        item_share = np.asarray(row.get("item_share_target", []), dtype=np.float32).reshape(-1)
        item_adv = np.asarray(row.get("item_adv_target", []), dtype=np.float32).reshape(-1)
        sid_share = np.asarray(row.get("sid_share_target", []), dtype=np.float32)
        sid_adv = np.asarray(row.get("sid_adv_target", []), dtype=np.float32)
        n_items = min(len(row["token_ids"]), int(item_share.shape[0]), int(item_adv.shape[0]), max_items)
        if n_items > 0:
            item_share_target[row_idx, :n_items] = torch.tensor(item_share[:n_items], dtype=torch.float32)
            item_adv_target[row_idx, :n_items] = torch.tensor(item_adv[:n_items], dtype=torch.float32)
        if sid_share.ndim == 2 and sid_share.shape[1] == sid_depth:
            n_sid_items = min(int(sid_share.shape[0]), max_items)
            sid_share_target[row_idx, :n_sid_items] = torch.tensor(sid_share[:n_sid_items], dtype=torch.float32)
        if sid_adv.ndim == 2 and sid_adv.shape[1] == sid_depth:
            n_sid_items = min(int(sid_adv.shape[0]), max_items)
            sid_adv_target[row_idx, :n_sid_items] = torch.tensor(sid_adv[:n_sid_items], dtype=torch.float32)
    if int(user_feat_dim) > 0:
        user_features = torch.stack(
            [
                torch.tensor(row.get("user_features", [0.0 for _ in range(int(user_feat_dim))]), dtype=torch.float32)
                for row in batch
            ],
            dim=0,
        )
    else:
        user_features = torch.zeros((len(batch), 0), dtype=torch.float32)
    return {
        "pre_input_ids": torch.stack([torch.tensor(x["pre_input_ids"], dtype=torch.long) for x in batch], dim=0),
        "pre_attention_mask": torch.stack([torch.tensor(x["pre_attention_mask"], dtype=torch.long) for x in batch], dim=0),
        "page_features": torch.stack([torch.tensor(x["page_features"], dtype=torch.float32) for x in batch], dim=0),
        "user_features": user_features,
        "token_ids": token_ids,
        "item_mask": item_mask,
        "q_target": torch.tensor([float(x["q_target"]) for x in batch], dtype=torch.float32),
        "item_share_target": item_share_target,
        "item_adv_target": item_adv_target,
        "sid_share_target": sid_share_target,
        "sid_adv_target": sid_adv_target,
        "groups": [str(x["group"]) for x in batch],
    }


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    masked_logits = logits.masked_fill(~mask, -1e9)
    probs = torch.softmax(masked_logits, dim=dim)
    probs = probs * mask.float()
    denom = probs.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    return probs / denom


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask_f = mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    return (values * mask_f).sum() / denom


def masked_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    beta: float = 1.0,
) -> torch.Tensor:
    beta = float(max(float(beta), 0.0))
    diff = (pred - target).abs()
    if beta <= 1e-8:
        loss = diff
    else:
        loss = torch.where(
            diff < beta,
            0.5 * diff.pow(2) / beta,
            diff - 0.5 * beta,
        )
    return masked_mean(loss, mask)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    return masked_mean((pred - target).abs(), mask)


def safe_correlation(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    if pred.numel() <= 1:
        return 0.0
    pred = pred.detach().float().reshape(-1)
    target = target.detach().float().reshape(-1)
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = float(pred.norm().item() * target.norm().item())
    if denom <= 1e-8:
        return 0.0
    return float(torch.dot(pred, target).item() / denom)
