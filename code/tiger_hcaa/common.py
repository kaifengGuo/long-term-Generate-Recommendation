import json
import random
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import utils  # noqa: E402
from reader import *  # noqa: F401,F403,E402

from tiger_phase2_blend_common import build_history_tokens  # noqa: E402


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


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_jsonl_rows(path: str | Path, max_rows: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as infile:
        for line_idx, line in enumerate(infile):
            if int(max_rows) > 0 and line_idx >= int(max_rows):
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


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


def weighted_response_strength(selected_response: Sequence[float], reward_weights: Sequence[float]) -> float:
    resp = np.asarray(list(selected_response), dtype=np.float32).reshape(-1)
    weights = np.asarray(list(reward_weights), dtype=np.float32).reshape(-1)
    width = min(int(resp.shape[0]), int(weights.shape[0]))
    if width <= 0:
        return 0.0
    return float(np.dot(resp[:width], weights[:width]))


def longest_prefix_ratio(hist_sid: np.ndarray, target_sid: np.ndarray, end_idx: int) -> float:
    width = int(end_idx) + 1
    if width <= 0:
        return 0.0
    match = float(np.mean(hist_sid[:width] == target_sid[:width]))
    return float(np.clip(match, 0.0, 1.0))


def compute_support_strength(history_sid: np.ndarray, target_tokens: Sequence[int]) -> float:
    target = np.asarray(list(target_tokens), dtype=np.int64).reshape(-1)
    if target.size == 0 or history_sid.size == 0:
        return 0.0
    affinity = np.zeros((history_sid.shape[0], target.shape[0]), dtype=np.float32)
    for ridx in range(history_sid.shape[0]):
        for cidx in range(target.shape[0]):
            affinity[ridx, cidx] = longest_prefix_ratio(history_sid[ridx], target, cidx)
    support_mean = float(np.mean(np.max(affinity, axis=0))) if affinity.size > 0 else 0.0
    support_full = float(np.max(affinity[:, -1])) if affinity.shape[1] > 0 else 0.0
    return float(0.5 * support_mean + 0.5 * support_full)


def history_sid_matrix(history_items: Sequence[int], iid2sid_tok_cpu: torch.Tensor) -> np.ndarray:
    hist = [int(x) for x in history_items if 0 <= int(x) < int(iid2sid_tok_cpu.shape[0])]
    if not hist:
        return np.zeros((0, int(iid2sid_tok_cpu.shape[1])), dtype=np.int64)
    hist_sid = iid2sid_tok_cpu[torch.tensor(hist, dtype=torch.long)].detach().cpu().numpy()
    valid = np.any(hist_sid > 0, axis=1)
    return hist_sid[valid]


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


def infer_post_history(rec: Dict[str, Any], max_append: int = 3) -> List[int]:
    history_items = [int(x) for x in rec.get("history_items", [])]
    selected_item_ids = [int(x) for x in rec.get("selected_item_ids", [])]
    item_rewards = np.asarray(rec.get("selected_item_rewards", []), dtype=np.float32).reshape(-1)
    responses = np.asarray(rec.get("selected_responses", []), dtype=np.float32)
    chosen: List[int] = []
    if item_rewards.ndim == 1 and item_rewards.shape[0] == len(selected_item_ids):
        chosen = [selected_item_ids[idx] for idx in range(len(selected_item_ids)) if float(item_rewards[idx]) > 0.0]
    if not chosen and responses.ndim == 2 and responses.shape[0] == len(selected_item_ids):
        strength = responses.sum(axis=1)
        chosen = [selected_item_ids[idx] for idx in range(len(selected_item_ids)) if float(strength[idx]) > 0.0]
    if not chosen:
        chosen = selected_item_ids[: max(int(max_append), 0)]
    return history_items + chosen


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


def build_page_features(
    *,
    page_index: int,
    max_page_index: int,
    history_len: int,
    max_hist_items: int,
    slate_size: int,
    max_slate_size: int,
    item_rewards: np.ndarray,
    response_strengths: np.ndarray,
    support_strengths: np.ndarray,
    step_reward: float,
    done: bool,
) -> np.ndarray:
    rewards = item_rewards if item_rewards.size > 0 else np.zeros((1,), dtype=np.float32)
    responses = response_strengths if response_strengths.size > 0 else np.zeros((1,), dtype=np.float32)
    supports = support_strengths if support_strengths.size > 0 else np.zeros((1,), dtype=np.float32)
    feat = np.asarray(
        [
            float(min(int(page_index), int(max_page_index)) / max(int(max_page_index), 1)),
            float(min(int(history_len), int(max_hist_items)) / max(int(max_hist_items), 1)),
            float(min(int(slate_size), int(max_slate_size)) / max(int(max_slate_size), 1)),
            float(step_reward),
            float(rewards.mean()),
            float(rewards.max()),
            float(rewards.min()),
            float(np.std(rewards)),
            float(np.mean(rewards > 0.0)),
            float(responses.mean()),
            float(responses.max()),
            float(responses.min()),
            float(supports.mean()),
            float(supports.max()),
            float(supports.min()),
            float(np.std(supports)),
            float(1.0 if bool(done) else 0.0),
        ],
        dtype=np.float32,
    )
    return feat


def build_item_features(
    *,
    slate_item_index: int,
    slate_size: int,
    item_reward: float,
    item_reward_centered: float,
    response_strength: float,
    response_strength_centered: float,
    support_strength: float,
    support_strength_centered: float,
    token_ids: Sequence[int],
    token_scale: float,
    done: bool,
) -> np.ndarray:
    token_feat = np.asarray(list(token_ids), dtype=np.float32).reshape(-1)
    if token_feat.size > 0 and float(token_scale) > 0.0:
        token_feat = token_feat / float(token_scale)
    feat = np.concatenate(
        [
            np.asarray(
                [
                    float(int(slate_item_index) / max(int(slate_size) - 1, 1)),
                    float(int(slate_size)),
                    float(item_reward),
                    float(item_reward_centered),
                    float(response_strength),
                    float(response_strength_centered),
                    float(support_strength),
                    float(support_strength_centered),
                    float(1.0 if bool(done) else 0.0),
                ],
                dtype=np.float32,
            ),
            token_feat.astype(np.float32),
        ],
        axis=0,
    )
    return feat.astype(np.float32)


def build_item_share_target(
    *,
    page_chain_rows: Sequence[Dict[str, Any]],
    slate_size: int,
    item_rewards: np.ndarray,
    response_strengths: np.ndarray,
) -> np.ndarray:
    first = page_chain_rows[0] if page_chain_rows else {}
    for key in [
        "selected_item_credit_shares_bootstrap",
        "selected_item_credit_shares",
        "selected_item_credit_shares_heuristic",
    ]:
        values = np.asarray(first.get(key, []), dtype=np.float32).reshape(-1)
        if values.shape[0] == int(slate_size):
            return normalize_probabilities(values, use_abs=True)
    reward_basis = np.maximum(item_rewards, 0.0)
    response_basis = np.maximum(response_strengths, 0.0)
    basis = reward_basis + 0.25 * response_basis
    return normalize_probabilities(basis, use_abs=False)


def build_token_share_targets(
    *,
    page_chain_rows: Sequence[Dict[str, Any]],
    selected_tokens_list: Sequence[Sequence[int]],
) -> np.ndarray:
    slate_size = len(selected_tokens_list)
    sid_depth = len(selected_tokens_list[0]) if selected_tokens_list else 0
    rows_by_item = {int(row.get("slate_item_index", -1)): row for row in page_chain_rows}
    out = np.zeros((slate_size, sid_depth), dtype=np.float32)
    for item_idx in range(slate_size):
        token_ids = [int(x) for x in selected_tokens_list[item_idx]]
        token_mask = np.asarray([int(x) > 0 for x in token_ids], dtype=bool)
        row = rows_by_item.get(int(item_idx), {})
        values: np.ndarray | None = None
        for key in ["token_credit_calibrated", "token_credit", "block_credit_calibrated", "block_credit"]:
            candidate = np.asarray(row.get(key, []), dtype=np.float32).reshape(-1)
            if candidate.shape[0] == int(sid_depth):
                values = candidate
                break
        if values is None:
            values = np.ones((sid_depth,), dtype=np.float32)
        out[item_idx] = normalize_probabilities(values, mask=token_mask, use_abs=True)
    return out


def build_episode_samples(
    *,
    trace_rows: Sequence[Dict[str, Any]],
    chain_rows: Sequence[Dict[str, Any]],
    iid2sid_tok_cpu: torch.Tensor,
    max_hist_items: int,
    gamma: float,
    hazard_lambda: float,
    max_episodes: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sid_depth = int(iid2sid_tok_cpu.shape[1])
    token_scale = float(max(int(iid2sid_tok_cpu.max().item()), 1))

    trace_grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in trace_rows:
        if "episode_id" not in rec or "selected_item_ids" not in rec:
            continue
        trace_grouped.setdefault(str(rec["episode_id"]), []).append(rec)
    episode_ids = sorted(trace_grouped.keys(), key=lambda x: int(x))
    if int(max_episodes) > 0:
        episode_ids = episode_ids[: int(max_episodes)]

    chain_grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in chain_rows:
        episode_id = str(row.get("episode_id", ""))
        page_index = int(row.get("page_index", -1))
        if episode_id and page_index >= 0:
            chain_grouped.setdefault((episode_id, page_index), []).append(row)

    max_page_index = 1
    max_slate_size = 1
    for episode_id in episode_ids:
        pages = trace_grouped.get(episode_id, [])
        for rec in pages:
            max_page_index = max(max_page_index, int(rec.get("page_index", 1)))
            max_slate_size = max(max_slate_size, len(rec.get("selected_item_ids", [])))

    page_rows: List[Dict[str, Any]] = []
    episodes: List[Dict[str, Any]] = []
    for episode_id in episode_ids:
        pages = sorted(trace_grouped.get(episode_id, []), key=lambda x: int(x.get("page_index", 0)))
        if not pages:
            continue
        lt_rewards = [
            float(rec.get("step_reward", 0.0)) - float(hazard_lambda) * float(1.0 if bool(rec.get("done", False)) else 0.0)
            for rec in pages
        ]
        pre_values = discounted_returns(lt_rewards, float(gamma))
        post_values = list(pre_values[1:]) + [0.0]
        episode_pages: List[Dict[str, Any]] = []

        for page_pos, rec in enumerate(pages):
            selected_item_ids = [int(x) for x in rec.get("selected_item_ids", [])]
            slate_size = len(selected_item_ids)
            if slate_size <= 0:
                continue
            selected_tokens_list = align_selected_tokens(
                selected_item_ids,
                rec.get("selected_sid_tokens_list", []),
                iid2sid_tok_cpu,
                int(sid_depth),
            )
            selected_responses = [[float(v) for v in row] for row in rec.get("selected_responses", [])]
            if len(selected_responses) != slate_size:
                width = len(selected_responses[0]) if selected_responses else len(rec.get("response_weights", []))
                selected_responses = [[0.0 for _ in range(width)] for _ in range(slate_size)]
            item_rewards = np.asarray(rec.get("selected_item_rewards", []), dtype=np.float32).reshape(-1)
            if item_rewards.shape[0] != slate_size:
                item_rewards = np.zeros((slate_size,), dtype=np.float32)
            reward_centered = item_rewards - float(item_rewards.mean()) if item_rewards.size > 0 else np.zeros((slate_size,), dtype=np.float32)

            history_items = [int(x) for x in rec.get("history_items", [])][-int(max_hist_items):]
            next_history = (
                [int(x) for x in pages[page_pos + 1].get("history_items", [])][-int(max_hist_items):]
                if page_pos + 1 < len(pages)
                else ([] if bool(rec.get("done", False)) else infer_post_history(rec)[-int(max_hist_items):])
            )
            pre_input_ids, pre_attention_mask = build_history_state(
                history_items,
                iid2sid_tok_cpu=iid2sid_tok_cpu,
                max_hist_items=int(max_hist_items),
                sid_depth=int(sid_depth),
            )
            post_input_ids, post_attention_mask = build_history_state(
                next_history,
                iid2sid_tok_cpu=iid2sid_tok_cpu,
                max_hist_items=int(max_hist_items),
                sid_depth=int(sid_depth),
            )

            history_sid = history_sid_matrix(history_items, iid2sid_tok_cpu)
            reward_weights = [float(x) for x in rec.get("response_weights", [])]
            response_strengths = np.asarray(
                [weighted_response_strength(resp, reward_weights) for resp in selected_responses],
                dtype=np.float32,
            )
            support_strengths = np.asarray(
                [compute_support_strength(history_sid, tokens) for tokens in selected_tokens_list],
                dtype=np.float32,
            )
            response_centered = (
                response_strengths - float(response_strengths.mean())
                if response_strengths.size > 0
                else np.zeros((slate_size,), dtype=np.float32)
            )
            support_centered = (
                support_strengths - float(support_strengths.mean())
                if support_strengths.size > 0
                else np.zeros((slate_size,), dtype=np.float32)
            )

            page_features = build_page_features(
                page_index=int(rec.get("page_index", page_pos + 1)),
                max_page_index=int(max_page_index),
                history_len=len(history_items),
                max_hist_items=int(max_hist_items),
                slate_size=int(slate_size),
                max_slate_size=int(max_slate_size),
                item_rewards=item_rewards,
                response_strengths=response_strengths,
                support_strengths=support_strengths,
                step_reward=float(rec.get("step_reward", 0.0)),
                done=bool(rec.get("done", False)),
            )
            item_features = np.stack(
                [
                    build_item_features(
                        slate_item_index=int(item_idx),
                        slate_size=int(slate_size),
                        item_reward=float(item_rewards[item_idx]) if item_idx < item_rewards.size else 0.0,
                        item_reward_centered=float(reward_centered[item_idx]) if item_idx < reward_centered.size else 0.0,
                        response_strength=float(response_strengths[item_idx]) if item_idx < response_strengths.size else 0.0,
                        response_strength_centered=float(response_centered[item_idx]) if item_idx < response_centered.size else 0.0,
                        support_strength=float(support_strengths[item_idx]) if item_idx < support_strengths.size else 0.0,
                        support_strength_centered=float(support_centered[item_idx]) if item_idx < support_centered.size else 0.0,
                        token_ids=selected_tokens_list[item_idx],
                        token_scale=float(token_scale),
                        done=bool(rec.get("done", False)),
                    )
                    for item_idx in range(slate_size)
                ],
                axis=0,
            ).astype(np.float32)

            page_chain_rows = sorted(
                chain_grouped.get((str(episode_id), int(rec.get("page_index", page_pos + 1))), []),
                key=lambda x: int(x.get("slate_item_index", 0)),
            )
            item_share_target = build_item_share_target(
                page_chain_rows=page_chain_rows,
                slate_size=int(slate_size),
                item_rewards=item_rewards,
                response_strengths=response_strengths,
            )
            token_share_target = build_token_share_targets(
                page_chain_rows=page_chain_rows,
                selected_tokens_list=selected_tokens_list,
            )

            page_row = {
                "episode_id": str(episode_id),
                "page_index": int(rec.get("page_index", page_pos + 1)),
                "pre_input_ids": pre_input_ids,
                "pre_attention_mask": pre_attention_mask,
                "post_input_ids": post_input_ids,
                "post_attention_mask": post_attention_mask,
                "page_features": page_features.tolist(),
                "item_features": item_features.tolist(),
                "token_ids": [list(tokens) for tokens in selected_tokens_list],
                "pre_value_target": float(pre_values[page_pos]),
                "post_value_target": float(post_values[page_pos]),
                "raw_delta_target": float(post_values[page_pos] - pre_values[page_pos]),
                "page_bias_target": 0.0,
                "page_adv_target": 0.0,
                "item_share_target": item_share_target.tolist(),
                "item_adv_target": [0.0 for _ in range(slate_size)],
                "token_share_target": token_share_target.tolist(),
                "token_adv_target": [[0.0 for _ in range(int(sid_depth))] for _ in range(slate_size)],
            }
            episode_pages.append(page_row)
            page_rows.append(page_row)

        if episode_pages:
            episodes.append({"group": str(episode_id), "pages": episode_pages})

    bias_by_page_index: Dict[int, float] = {}
    bucket: Dict[int, List[float]] = {}
    for row in page_rows:
        bucket.setdefault(int(row["page_index"]), []).append(float(row["raw_delta_target"]))
    for page_index, values in bucket.items():
        bias_by_page_index[int(page_index)] = float(np.mean(values)) if values else 0.0

    for episode in episodes:
        pages = episode["pages"]
        bias_seq = np.asarray([bias_by_page_index.get(int(row["page_index"]), 0.0) for row in pages], dtype=np.float32)
        if bias_seq.size > 0:
            bias_seq = bias_seq - float(bias_seq.mean())
        for row_idx, row in enumerate(pages):
            page_bias = float(bias_seq[row_idx]) if row_idx < bias_seq.size else 0.0
            page_adv = float(row["raw_delta_target"]) - page_bias
            item_share = np.asarray(row["item_share_target"], dtype=np.float32).reshape(-1)
            token_share = np.asarray(row["token_share_target"], dtype=np.float32)
            item_adv = (item_share * float(page_adv)).astype(np.float32)
            token_adv = (token_share * item_adv.reshape(-1, 1)).astype(np.float32)
            row["page_bias_target"] = float(page_bias)
            row["page_adv_target"] = float(page_adv)
            row["item_adv_target"] = item_adv.tolist()
            row["token_adv_target"] = token_adv.tolist()

    samples: List[Dict[str, Any]] = []
    page_dim = 0
    item_dim = 0
    for episode in episodes:
        pages = episode["pages"]
        page_dim = max(page_dim, len(pages[0]["page_features"]))
        item_dim = max(item_dim, len(pages[0]["item_features"][0]))
        samples.append(
            {
                "group": str(episode["group"]),
                "n_pages": len(pages),
                "page_index": [int(row["page_index"]) for row in pages],
                "pre_input_ids": [row["pre_input_ids"] for row in pages],
                "pre_attention_mask": [row["pre_attention_mask"] for row in pages],
                "post_input_ids": [row["post_input_ids"] for row in pages],
                "post_attention_mask": [row["post_attention_mask"] for row in pages],
                "page_features": [row["page_features"] for row in pages],
                "item_features": [row["item_features"] for row in pages],
                "token_ids": [row["token_ids"] for row in pages],
                "pre_value_target": [row["pre_value_target"] for row in pages],
                "post_value_target": [row["post_value_target"] for row in pages],
                "raw_delta_target": [row["raw_delta_target"] for row in pages],
                "page_bias_target": [row["page_bias_target"] for row in pages],
                "page_adv_target": [row["page_adv_target"] for row in pages],
                "item_share_target": [row["item_share_target"] for row in pages],
                "item_adv_target": [row["item_adv_target"] for row in pages],
                "token_share_target": [row["token_share_target"] for row in pages],
                "token_adv_target": [row["token_adv_target"] for row in pages],
            }
        )

    meta = {
        "n_trace_rows": int(len(trace_rows)),
        "n_chain_rows": int(len(chain_rows)),
        "n_episodes": int(len(samples)),
        "page_dim": int(page_dim),
        "item_dim": int(item_dim),
        "sid_depth": int(sid_depth),
        "max_page_index": int(max_page_index),
        "max_slate_size": int(max_slate_size),
        "gamma": float(gamma),
        "hazard_lambda": float(hazard_lambda),
        "page_bias_by_index": {str(k): float(v) for k, v in sorted(bias_by_page_index.items())},
    }
    return samples, meta


class EpisodeDataset(torch.utils.data.Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[int(idx)]


def collate_episodes(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    batch_size = len(batch)
    max_pages = max(int(x["n_pages"]) for x in batch)
    hist_len = len(batch[0]["pre_input_ids"][0])
    page_dim = len(batch[0]["page_features"][0])
    token_len = len(batch[0]["token_ids"][0][0])
    max_items = max(len(page_items) for sample in batch for page_items in sample["token_ids"])
    item_dim = len(batch[0]["item_features"][0][0])

    pre_input_ids = torch.zeros((batch_size, max_pages, hist_len), dtype=torch.long)
    pre_attention_mask = torch.zeros((batch_size, max_pages, hist_len), dtype=torch.long)
    post_input_ids = torch.zeros((batch_size, max_pages, hist_len), dtype=torch.long)
    post_attention_mask = torch.zeros((batch_size, max_pages, hist_len), dtype=torch.long)
    page_features = torch.zeros((batch_size, max_pages, page_dim), dtype=torch.float32)
    item_features = torch.zeros((batch_size, max_pages, max_items, item_dim), dtype=torch.float32)
    token_ids = torch.zeros((batch_size, max_pages, max_items, token_len), dtype=torch.long)
    page_mask = torch.zeros((batch_size, max_pages), dtype=torch.bool)
    item_mask = torch.zeros((batch_size, max_pages, max_items), dtype=torch.bool)
    token_mask = torch.zeros((batch_size, max_pages, max_items, token_len), dtype=torch.bool)

    pre_value_target = torch.zeros((batch_size, max_pages), dtype=torch.float32)
    post_value_target = torch.zeros((batch_size, max_pages), dtype=torch.float32)
    raw_delta_target = torch.zeros((batch_size, max_pages), dtype=torch.float32)
    page_bias_target = torch.zeros((batch_size, max_pages), dtype=torch.float32)
    page_adv_target = torch.zeros((batch_size, max_pages), dtype=torch.float32)
    item_share_target = torch.zeros((batch_size, max_pages, max_items), dtype=torch.float32)
    item_adv_target = torch.zeros((batch_size, max_pages, max_items), dtype=torch.float32)
    token_share_target = torch.zeros((batch_size, max_pages, max_items, token_len), dtype=torch.float32)
    token_adv_target = torch.zeros((batch_size, max_pages, max_items, token_len), dtype=torch.float32)
    page_index = torch.zeros((batch_size, max_pages), dtype=torch.long)
    groups: List[str] = []

    for batch_idx, sample in enumerate(batch):
        groups.append(str(sample["group"]))
        n_pages = int(sample["n_pages"])
        for page_idx in range(n_pages):
            n_items = len(sample["token_ids"][page_idx])
            page_mask[batch_idx, page_idx] = True
            pre_input_ids[batch_idx, page_idx] = torch.tensor(sample["pre_input_ids"][page_idx], dtype=torch.long)
            pre_attention_mask[batch_idx, page_idx] = torch.tensor(sample["pre_attention_mask"][page_idx], dtype=torch.long)
            post_input_ids[batch_idx, page_idx] = torch.tensor(sample["post_input_ids"][page_idx], dtype=torch.long)
            post_attention_mask[batch_idx, page_idx] = torch.tensor(sample["post_attention_mask"][page_idx], dtype=torch.long)
            page_features[batch_idx, page_idx] = torch.tensor(sample["page_features"][page_idx], dtype=torch.float32)
            pre_value_target[batch_idx, page_idx] = float(sample["pre_value_target"][page_idx])
            post_value_target[batch_idx, page_idx] = float(sample["post_value_target"][page_idx])
            raw_delta_target[batch_idx, page_idx] = float(sample["raw_delta_target"][page_idx])
            page_bias_target[batch_idx, page_idx] = float(sample["page_bias_target"][page_idx])
            page_adv_target[batch_idx, page_idx] = float(sample["page_adv_target"][page_idx])
            page_index[batch_idx, page_idx] = int(sample["page_index"][page_idx])

            item_mask[batch_idx, page_idx, :n_items] = True
            item_features[batch_idx, page_idx, :n_items] = torch.tensor(sample["item_features"][page_idx], dtype=torch.float32)
            item_share_target[batch_idx, page_idx, :n_items] = torch.tensor(sample["item_share_target"][page_idx], dtype=torch.float32)
            item_adv_target[batch_idx, page_idx, :n_items] = torch.tensor(sample["item_adv_target"][page_idx], dtype=torch.float32)
            token_ids[batch_idx, page_idx, :n_items] = torch.tensor(sample["token_ids"][page_idx], dtype=torch.long)
            token_share_target[batch_idx, page_idx, :n_items] = torch.tensor(sample["token_share_target"][page_idx], dtype=torch.float32)
            token_adv_target[batch_idx, page_idx, :n_items] = torch.tensor(sample["token_adv_target"][page_idx], dtype=torch.float32)
            token_mask[batch_idx, page_idx, :n_items] = token_ids[batch_idx, page_idx, :n_items] > 0

    return {
        "groups": groups,
        "page_index": page_index,
        "page_mask": page_mask,
        "item_mask": item_mask,
        "token_mask": token_mask,
        "pre_input_ids": pre_input_ids,
        "pre_attention_mask": pre_attention_mask,
        "post_input_ids": post_input_ids,
        "post_attention_mask": post_attention_mask,
        "page_features": page_features,
        "item_features": item_features,
        "token_ids": token_ids,
        "pre_value_target": pre_value_target,
        "post_value_target": post_value_target,
        "raw_delta_target": raw_delta_target,
        "page_bias_target": page_bias_target,
        "page_adv_target": page_adv_target,
        "item_share_target": item_share_target,
        "item_adv_target": item_adv_target,
        "token_share_target": token_share_target,
        "token_adv_target": token_adv_target,
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


def masked_huber_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return masked_mean(loss, mask)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    return masked_mean((pred - target).abs(), mask)


def masked_kl_div(pred_probs: torch.Tensor, target_probs: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    pred = pred_probs.clamp_min(1e-8) * mask.float()
    pred = pred / pred.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    target = target_probs.clamp_min(1e-8) * mask.float()
    target = target / target.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    kl = torch.where(mask, target * (target.log() - pred.log()), torch.zeros_like(target))
    group_mask = mask.any(dim=dim)
    return masked_mean(kl.sum(dim=dim), group_mask)


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
