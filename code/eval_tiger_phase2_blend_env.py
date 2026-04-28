# -*- coding: utf-8 -*-
"""
Evaluate TIGER Base + Phase2 Blend Decode in KuaiSim.

This script does not modify the original eval_tiger_env.py. It provides:
1. a standalone blend-decode evaluator;
2. optional trace dumping for Phase2 head training.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from env.KREnvironment_WholeSession_GPU import KREnvironment_WholeSession_GPU
from model.reward import get_immediate_reward
import utils

from tiger_phase2_blend_common import (
    TokenCreditTransportHead,
    TokenPrefixValueHead,
    TokenLongTermActorHead,
    build_history_tokens,
    build_iid2sid_tokens,
    build_sid_prefix_to_next,
    infer_model_size_args,
    load_phase2_head,
    load_prefix_value_head,
    load_phase3_actor_head,
    load_tiger_model,
)
from tiger_slate_online_common import (
    OnlineSlateAllocatorHead,
    SlateValueHead,
    build_online_slate_inputs,
    load_online_slate_allocator,
    load_slate_value_head,
)
from tiger_phase5_token_actor_common import (
    TokenResidualActorHead,
    load_phase5_token_actor_head,
)
from tiger_phase6_joint_common import (
    HistoryPlanHead,
    PlanConditionedPrefixValueHead,
    PlanConditionedTokenActorHead,
    load_phase6_joint_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TigerPhase2BlendPolicy:
    def __init__(
        self,
        *,
        model,
        iid2sid_tok: torch.Tensor,
        sid2iid_map_tok: Dict[Tuple[int, ...], int],
        sid_depth: int,
        device: torch.device,
        slate_size: int,
        reader,
        args,
        phase2_head: TokenCreditTransportHead | None,
        phase4_prefix_head: TokenPrefixValueHead | None,
        phase3_actor_head: TokenLongTermActorHead | None,
        phase5_token_actor_head: TokenResidualActorHead | None,
        phase6_plan_head: HistoryPlanHead | None,
        phase6_prefix_head: PlanConditionedPrefixValueHead | None,
        phase6_token_actor_head: PlanConditionedTokenActorHead | None,
        online_slate_allocator_head: OnlineSlateAllocatorHead | None,
        slate_value_head: SlateValueHead | None,
    ):
        self.model = model
        self.iid2sid_tok = iid2sid_tok
        self.iid2sid_tok_cpu = iid2sid_tok.detach().cpu()
        self.sid2iid_map = sid2iid_map_tok
        self.sid_depth = int(sid_depth)
        self.device = device
        self.slate_size = int(slate_size)
        self.reader = reader
        self.beam_width = int(args.beam_width)
        self.max_hist_items = int(args.max_hist_items)
        self.blend_scale = float(args.phase2_blend_scale)
        self.decode_topk = int(args.phase2_decode_topk)
        self.phase2_head = phase2_head
        self.phase4_prefix_head = phase4_prefix_head
        self.phase4_prefix_scale = float(getattr(args, "phase4_prefix_scale", 0.0))
        self.phase3_actor_head = phase3_actor_head
        self.actor_scale = float(getattr(args, "phase3_actor_scale", 0.0))
        self.phase5_token_actor_head = phase5_token_actor_head
        self.phase5_actor_scale = float(getattr(args, "phase5_token_actor_scale", 0.0))
        self.phase6_plan_head = phase6_plan_head
        self.phase6_prefix_head = phase6_prefix_head
        self.phase6_prefix_scale = float(getattr(args, "phase6_prefix_scale", 0.0))
        self.phase6_token_actor_head = phase6_token_actor_head
        self.phase6_token_actor_scale = float(getattr(args, "phase6_token_actor_scale", 0.0))
        self.online_slate_allocator_head = online_slate_allocator_head
        self.online_slate_allocator_scale = float(getattr(args, "online_slate_allocator_scale", 0.0))
        self.slate_value_head = slate_value_head
        self.slate_value_scale = float(getattr(args, "slate_value_scale", 0.0))
        self.slate_rerank_pool = int(getattr(args, "slate_rerank_pool", max(self.slate_size, 8)))
        self.slate_greedy_candidates = int(getattr(args, "slate_greedy_candidates", max(self.slate_size, 6)))
        self.slate_full_rerank = bool(getattr(args, "slate_full_rerank", False))
        self.slate_num_candidate_slates = int(getattr(args, "slate_num_candidate_slates", 6))
        self.sid_prefix_to_next = build_sid_prefix_to_next(self.sid2iid_map)
        self.fast_base_generate = bool(getattr(args, "fast_base_generate", False))
        self.use_decoder_kv_cache = not bool(getattr(args, "disable_decoder_kv_cache", False))
        self.token_vocab_size = int(self.iid2sid_tok_cpu.max().item()) + 1
        self.random_topk_sample = int(getattr(args, "random_topk_sample", 0))
        self.random_item_prob = float(getattr(args, "random_item_prob", 0.0))
        self.last_action_debug: List[Dict[str, Any]] = []

    def _allowed_next_tokens(self, prefix: List[int]) -> List[int]:
        return self.sid_prefix_to_next.get(tuple(int(x) for x in prefix), [])

    def _compute_phase6_plan_probs(self, encoder_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor | None:
        if self.phase6_plan_head is None:
            return None
        mask = attention_mask.unsqueeze(-1).float().to(encoder_hidden.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        summary = (encoder_hidden * mask).sum(dim=1) / denom
        logits, _value = self.phase6_plan_head(summary)
        return torch.softmax(logits, dim=-1)

    def _decode_ranked_sequences(
        self,
        encoder_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        phase6_plan_probs: torch.Tensor | None = None,
    ) -> List[Tuple[List[int], float]]:
        if not self.use_decoder_kv_cache:
            return self._decode_ranked_sequences_no_cache(
                encoder_hidden=encoder_hidden,
                attention_mask=attention_mask,
                phase6_plan_probs=phase6_plan_probs,
            )
        return self._decode_ranked_sequences_kv_cache(
            encoder_hidden=encoder_hidden,
            attention_mask=attention_mask,
            phase6_plan_probs=phase6_plan_probs,
        )

    def _decode_ranked_sequences_no_cache(
        self,
        encoder_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        phase6_plan_probs: torch.Tensor | None = None,
    ) -> List[Tuple[List[int], float]]:
        beams: List[Tuple[List[int], float]] = [([], 0.0)]
        for _step in range(self.sid_depth):
            prefixes = [[0] + seq for seq, _score in beams]
            max_len = max(len(p) for p in prefixes)
            dec = np.zeros((len(prefixes), max_len), dtype=np.int64)
            for row_idx, prefix in enumerate(prefixes):
                dec[row_idx, -len(prefix):] = np.asarray(prefix, dtype=np.int64)

            dec_input = torch.tensor(dec, dtype=torch.long, device=self.device)
            enc_hidden = encoder_hidden.expand(len(prefixes), -1, -1)
            enc_attn = attention_mask.repeat(len(prefixes), 1)
            logits, hidden = self.model.decode_with_hidden_from_encoded(
                encoder_hidden=enc_hidden,
                attention_mask=enc_attn,
                decoder_input_ids=dec_input,
            )
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
            if self.phase2_head is not None and self.blend_scale != 0.0:
                credit_logits = self.phase2_head.score_all_tokens(hidden[:, -1, :])
                step_scores = log_probs + float(self.blend_scale) * credit_logits
            else:
                step_scores = log_probs
            if self.phase4_prefix_head is not None and self.phase4_prefix_scale != 0.0:
                prefix_logits = self.phase4_prefix_head.score_all_tokens(hidden[:, -1, :])
                prefix_logits = prefix_logits - prefix_logits.mean(dim=-1, keepdim=True)
                step_scores = step_scores + float(self.phase4_prefix_scale) * prefix_logits
            if self.phase3_actor_head is not None and self.actor_scale != 0.0:
                actor_logits = self.phase3_actor_head(hidden[:, -1, :])
                actor_log_probs = torch.log_softmax(actor_logits, dim=-1)
                step_scores = step_scores + float(self.actor_scale) * actor_log_probs
            if self.phase5_token_actor_head is not None and self.phase5_actor_scale != 0.0:
                residual_scores = self.phase5_token_actor_head.score_all_tokens(hidden[:, -1, :])
                residual_scores = residual_scores - residual_scores.mean(dim=-1, keepdim=True)
                step_scores = step_scores + float(self.phase5_actor_scale) * residual_scores
            if phase6_plan_probs is not None and self.phase6_prefix_head is not None and self.phase6_prefix_scale != 0.0:
                phase6_plan_expand = phase6_plan_probs.expand(hidden.shape[0], -1)
                phase6_prefix_logits = self.phase6_prefix_head.score_all_tokens(hidden[:, -1, :], phase6_plan_expand)
                phase6_prefix_logits = phase6_prefix_logits - phase6_prefix_logits.mean(dim=-1, keepdim=True)
                step_scores = step_scores + float(self.phase6_prefix_scale) * phase6_prefix_logits
            if phase6_plan_probs is not None and self.phase6_token_actor_head is not None and self.phase6_token_actor_scale != 0.0:
                phase6_plan_expand = phase6_plan_probs.expand(hidden.shape[0], -1)
                phase6_actor_scores = self.phase6_token_actor_head.score_all_tokens(hidden[:, -1, :], phase6_plan_expand)
                phase6_actor_scores = phase6_actor_scores - phase6_actor_scores.mean(dim=-1, keepdim=True)
                step_scores = step_scores + float(self.phase6_token_actor_scale) * phase6_actor_scores
            step_scores[:, 0] = -1e9

            candidates: List[Tuple[List[int], float]] = []
            for beam_idx, (seq, base_score) in enumerate(beams):
                allowed = self._allowed_next_tokens(seq)
                if not allowed:
                    continue
                allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=self.device)
                allowed_scores = step_scores[beam_idx, allowed_tensor]
                topk_cfg = int(self.decode_topk) if int(self.decode_topk) > 0 else int(self.beam_width)
                topk = min(topk_cfg, int(allowed_scores.shape[0]), int(self.beam_width))
                vals, idxs = torch.topk(allowed_scores, k=topk, dim=0)
                for local_pos in range(topk):
                    tok = int(allowed[int(idxs[local_pos].item())])
                    score = float(base_score + vals[local_pos].item())
                    candidates.append((seq + [tok], score))

            if not candidates:
                break
            candidates.sort(key=lambda x: x[1], reverse=True)
            uniq: List[Tuple[List[int], float]] = []
            seen = set()
            for seq, score in candidates:
                key = tuple(seq)
                if key in seen:
                    continue
                seen.add(key)
                uniq.append((seq, score))
                if len(uniq) >= int(self.beam_width):
                    break
            beams = uniq
        return beams

    def _decode_ranked_sequences_kv_cache(
        self,
        encoder_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        phase6_plan_probs: torch.Tensor | None = None,
    ) -> List[Tuple[List[int], float]]:
        beams: List[Tuple[List[int], float]] = [([], 0.0)]
        next_input_tokens = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        past_key_values = None

        for _step in range(self.sid_depth):
            n_beams = len(beams)
            enc_hidden = encoder_hidden.expand(n_beams, -1, -1)
            enc_attn = attention_mask.expand(n_beams, -1)
            logits, hidden, next_past = self.model.decode_step_with_cache_from_encoded(
                encoder_hidden=enc_hidden,
                attention_mask=enc_attn,
                decoder_input_ids=next_input_tokens,
                past_key_values=past_key_values,
            )
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
            if self.phase2_head is not None and self.blend_scale != 0.0:
                credit_logits = self.phase2_head.score_all_tokens(hidden[:, -1, :])
                step_scores = log_probs + float(self.blend_scale) * credit_logits
            else:
                step_scores = log_probs
            if self.phase4_prefix_head is not None and self.phase4_prefix_scale != 0.0:
                prefix_logits = self.phase4_prefix_head.score_all_tokens(hidden[:, -1, :])
                prefix_logits = prefix_logits - prefix_logits.mean(dim=-1, keepdim=True)
                step_scores = step_scores + float(self.phase4_prefix_scale) * prefix_logits
            if self.phase3_actor_head is not None and self.actor_scale != 0.0:
                actor_logits = self.phase3_actor_head(hidden[:, -1, :])
                actor_log_probs = torch.log_softmax(actor_logits, dim=-1)
                step_scores = step_scores + float(self.actor_scale) * actor_log_probs
            if self.phase5_token_actor_head is not None and self.phase5_actor_scale != 0.0:
                residual_scores = self.phase5_token_actor_head.score_all_tokens(hidden[:, -1, :])
                residual_scores = residual_scores - residual_scores.mean(dim=-1, keepdim=True)
                step_scores = step_scores + float(self.phase5_actor_scale) * residual_scores
            if phase6_plan_probs is not None and self.phase6_prefix_head is not None and self.phase6_prefix_scale != 0.0:
                phase6_plan_expand = phase6_plan_probs.expand(hidden.shape[0], -1)
                phase6_prefix_logits = self.phase6_prefix_head.score_all_tokens(hidden[:, -1, :], phase6_plan_expand)
                phase6_prefix_logits = phase6_prefix_logits - phase6_prefix_logits.mean(dim=-1, keepdim=True)
                step_scores = step_scores + float(self.phase6_prefix_scale) * phase6_prefix_logits
            if phase6_plan_probs is not None and self.phase6_token_actor_head is not None and self.phase6_token_actor_scale != 0.0:
                phase6_plan_expand = phase6_plan_probs.expand(hidden.shape[0], -1)
                phase6_actor_scores = self.phase6_token_actor_head.score_all_tokens(hidden[:, -1, :], phase6_plan_expand)
                phase6_actor_scores = phase6_actor_scores - phase6_actor_scores.mean(dim=-1, keepdim=True)
                step_scores = step_scores + float(self.phase6_token_actor_scale) * phase6_actor_scores
            step_scores[:, 0] = -1e9

            candidates: List[Tuple[int, List[int], float, int]] = []
            for beam_idx, (seq, base_score) in enumerate(beams):
                allowed = self._allowed_next_tokens(seq)
                if not allowed:
                    continue
                allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=self.device)
                allowed_scores = step_scores[beam_idx, allowed_tensor]
                topk_cfg = int(self.decode_topk) if int(self.decode_topk) > 0 else int(self.beam_width)
                topk = min(topk_cfg, int(allowed_scores.shape[0]), int(self.beam_width))
                vals, idxs = torch.topk(allowed_scores, k=topk, dim=0)
                for local_pos in range(topk):
                    tok = int(allowed[int(idxs[local_pos].item())])
                    score = float(base_score + vals[local_pos].item())
                    candidates.append((int(beam_idx), seq + [tok], score, tok))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[2], reverse=True)
            uniq: List[Tuple[List[int], float]] = []
            next_parent_indices: List[int] = []
            next_tokens: List[int] = []
            seen = set()
            for parent_idx, seq, score, tok in candidates:
                key = tuple(seq)
                if key in seen:
                    continue
                seen.add(key)
                uniq.append((seq, score))
                next_parent_indices.append(int(parent_idx))
                next_tokens.append(int(tok))
                if len(uniq) >= int(self.beam_width):
                    break
            beams = uniq

            if not beams or _step >= int(self.sid_depth) - 1:
                continue

            beam_idx_tensor = torch.tensor(next_parent_indices, dtype=torch.long, device=self.device)
            past_key_values = self.model.reorder_cache(next_past, beam_idx_tensor)
            next_input_tokens = torch.tensor(next_tokens, dtype=torch.long, device=self.device).unsqueeze(-1)

        return beams

    @torch.no_grad()
    def act(self, observation, candidate_info):
        hist_iids = observation["user_history"]["history"].long().to(self.device)
        cand_iids = candidate_info["item_id"].long().detach().cpu().numpy()
        global_to_local = {int(gid): idx for idx, gid in enumerate(cand_iids)}
        input_ids, attention_mask = build_history_tokens(
            hist_iids,
            self.iid2sid_tok,
            int(self.max_hist_items),
            int(self.sid_depth),
        )
        bsz = int(hist_iids.size(0))
        actions: List[List[int]] = []
        action_debug: List[Dict[str, Any]] = []

        use_fast_base = (
            self.fast_base_generate
            and self.phase2_head is None
            and self.phase4_prefix_head is None
            and self.phase3_actor_head is None
            and self.phase5_token_actor_head is None
            and self.phase6_plan_head is None
            and self.phase6_prefix_head is None
            and self.phase6_token_actor_head is None
        )
        if use_fast_base:
            gen = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=int(self.beam_width),
                num_return_sequences=int(self.beam_width),
                max_length=int(self.sid_depth) + 1,
                early_stopping=True,
                do_sample=False,
            )
            gen = gen[:, 1:1 + int(self.sid_depth)]
            gen = gen.view(bsz, int(self.beam_width), int(self.sid_depth))
            gen_np = gen.detach().cpu().numpy()

        for row_idx in range(bsz):
            user_hist_items = set(int(x) for x in hist_iids[row_idx].detach().cpu().numpy().tolist())
            item_score_pairs: List[Tuple[int, float]] = []
            if use_fast_base:
                beams = gen_np[row_idx]
                for rank in range(int(self.beam_width)):
                    key = tuple(int(x) for x in beams[rank].tolist())
                    gid = self.sid2iid_map.get(key)
                    if gid is None:
                        continue
                    if gid not in global_to_local or gid in user_hist_items:
                        continue
                    score = float(int(self.beam_width) - rank)
                    item_score_pairs.append((int(global_to_local[gid]), score))
            else:
                encoder_hidden = self.model.encode(
                    input_ids[row_idx:row_idx + 1],
                    attention_mask[row_idx:row_idx + 1],
                )
                phase6_plan_probs = self._compute_phase6_plan_probs(
                    encoder_hidden,
                    attention_mask[row_idx:row_idx + 1],
                )
                ranked_sequences = self._decode_ranked_sequences(
                    encoder_hidden,
                    attention_mask[row_idx:row_idx + 1],
                    phase6_plan_probs=phase6_plan_probs,
                )
                for rank, (sid_seq, score) in enumerate(ranked_sequences):
                    key = tuple(int(x) for x in sid_seq)
                    gid = self.sid2iid_map.get(key)
                    if gid is None:
                        continue
                    if gid not in global_to_local or gid in user_hist_items:
                        continue
                    item_score_pairs.append((int(global_to_local[gid]), float(score - 1e-4 * rank)))

            item_score_pairs.sort(key=lambda x: x[1], reverse=True)
            score_by_idx: Dict[int, float] = {}
            candidate_idxs: List[int] = []
            for idx_local, score in item_score_pairs:
                if int(idx_local) in score_by_idx:
                    continue
                score_by_idx[int(idx_local)] = float(score)
                candidate_idxs.append(int(idx_local))

            if self.slate_size > 1 and candidate_idxs and (self.online_slate_allocator_head is not None or self.slate_value_head is not None):
                reranked = self._rerank_slate_from_online_heads(
                    hist_iids[row_idx],
                    cand_iids,
                    candidate_idxs,
                    score_by_idx,
                )
                if reranked:
                    candidate_idxs = reranked

            greedy_candidate_idxs = [int(x) for x in candidate_idxs]
            selection_mode = "greedy"
            topk_pool_size = 0
            sampled_ranks: List[int] = []
            sampled_local_pre_fill: List[int] = []

            if float(self.random_item_prob) > 0.0 and np.random.rand() < float(self.random_item_prob):
                available_random = [
                    int(idx_local)
                    for idx_local, gid in enumerate(cand_iids.tolist())
                    if int(gid) not in user_hist_items
                ]
                if available_random:
                    take = min(int(self.slate_size), int(len(available_random)))
                    sampled = np.random.choice(
                        np.asarray(available_random, dtype=np.int64),
                        size=take,
                        replace=False,
                    ).tolist()
                    candidate_idxs = [int(x) for x in sampled]
                    selection_mode = "random_item"
                    sampled_local_pre_fill = [int(x) for x in candidate_idxs]
                    sampled_ranks = [-1 for _ in sampled_local_pre_fill]
            elif int(self.random_topk_sample) > 0 and greedy_candidate_idxs:
                topk_pool_size = min(
                    max(int(self.random_topk_sample), int(self.slate_size)),
                    int(len(greedy_candidate_idxs)),
                )
                topk_pool = [int(x) for x in greedy_candidate_idxs[:topk_pool_size]]
                take = min(int(self.slate_size), int(len(topk_pool)))
                if take > 0:
                    sampled = np.random.choice(
                        np.asarray(topk_pool, dtype=np.int64),
                        size=take,
                        replace=False,
                    ).tolist()
                    candidate_idxs = [int(x) for x in sampled]
                    selection_mode = "topk_uniform"
                    sampled_local_pre_fill = [int(x) for x in candidate_idxs]
                    sampled_ranks = [int(topk_pool.index(int(x))) for x in sampled_local_pre_fill]

            if len(candidate_idxs) < int(self.slate_size):
                need = int(self.slate_size) - len(candidate_idxs)
                all_idx = np.arange(len(cand_iids))
                remain = np.setdiff1d(all_idx, np.asarray(candidate_idxs, dtype=np.int64))
                if len(remain) > 0:
                    take = min(int(need), int(len(remain)))
                    fill = np.random.choice(remain, size=take, replace=False).tolist()
                    candidate_idxs.extend(int(x) for x in fill)

            final_action = [int(x) for x in candidate_idxs[: int(self.slate_size)]]
            actions.append(final_action)
            action_debug.append(
                {
                    "selection_mode": str(selection_mode),
                    "selection_topk": int(topk_pool_size),
                    "selection_random_item_prob": float(self.random_item_prob),
                    "selection_sampled_ranks": [int(x) for x in sampled_ranks],
                    "selection_greedy_local": [
                        int(x)
                        for x in greedy_candidate_idxs[: max(int(self.random_topk_sample), int(self.slate_size), 1)]
                    ],
                    "selection_sampled_local_pre_fill": [int(x) for x in sampled_local_pre_fill],
                    "selection_final_local": [int(x) for x in final_action],
                }
            )
        self.last_action_debug = action_debug
        return torch.tensor(actions, dtype=torch.long, device=self.device)

    def _score_slate_value(
        self,
        history_items: Sequence[int],
        cand_iids: np.ndarray,
        slate_candidate_idxs: Sequence[int],
        base_scores: Sequence[float],
    ) -> float:
        if self.slate_value_head is None or not slate_candidate_idxs:
            return 0.0
        global_ids = [int(cand_iids[int(idx)]) for idx in slate_candidate_idxs if 0 <= int(idx) < len(cand_iids)]
        sid_tokens_list = []
        for gid in global_ids:
            if 0 <= int(gid) < int(self.iid2sid_tok.shape[0]):
                sid_tokens_list.append([int(x) for x in self.iid2sid_tok[int(gid)].detach().cpu().tolist()])
            else:
                sid_tokens_list.append([])
        online = build_online_slate_inputs(
            history_items=[int(x) for x in history_items],
            candidate_item_ids=global_ids,
            candidate_sid_tokens_list=sid_tokens_list,
            iid2sid_tok_cpu=self.iid2sid_tok_cpu,
            max_hist_items=int(self.max_hist_items),
            token_vocab_size=int(self.token_vocab_size),
            base_scores=[float(x) for x in base_scores],
        )
        item_features = torch.tensor(online["item_features"], dtype=torch.float32, device=self.device).unsqueeze(0)
        page_features = torch.tensor(online["page_features"], dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.ones((1, item_features.shape[1]), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            value = self.slate_value_head(item_features, page_features, mask=mask)
        return float(value.squeeze(0).item())

    def _rerank_slate_from_online_heads(
        self,
        hist_row: torch.Tensor,
        cand_iids: np.ndarray,
        candidate_idxs: Sequence[int],
        score_by_idx: Dict[int, float],
    ) -> List[int]:
        candidate_pool = [int(x) for x in candidate_idxs[: max(int(self.slate_rerank_pool), int(self.slate_size))]]
        if len(candidate_pool) <= int(self.slate_size):
            return list(candidate_pool)

        history_items = [int(x) for x in hist_row.detach().cpu().numpy().tolist() if int(x) > 0]
        global_ids = [int(cand_iids[int(idx)]) for idx in candidate_pool]
        sid_tokens_list = []
        base_scores = [float(score_by_idx.get(int(idx), 0.0)) for idx in candidate_pool]
        for gid in global_ids:
            if 0 <= int(gid) < int(self.iid2sid_tok.shape[0]):
                sid_tokens_list.append([int(x) for x in self.iid2sid_tok[int(gid)].detach().cpu().tolist()])
            else:
                sid_tokens_list.append([])

        online = build_online_slate_inputs(
            history_items=history_items,
            candidate_item_ids=global_ids,
            candidate_sid_tokens_list=sid_tokens_list,
            iid2sid_tok_cpu=self.iid2sid_tok_cpu,
            max_hist_items=int(self.max_hist_items),
            token_vocab_size=int(self.token_vocab_size),
            base_scores=base_scores,
        )
        base_arr = np.asarray(base_scores, dtype=np.float32)
        if base_arr.size > 0 and float(np.std(base_arr)) > 1e-8:
            base_norm = (base_arr - float(np.mean(base_arr))) / float(np.std(base_arr))
        else:
            base_norm = np.zeros_like(base_arr, dtype=np.float32)

        alloc_bonus = np.zeros_like(base_arr, dtype=np.float32)
        if self.online_slate_allocator_head is not None:
            item_features = torch.tensor(online["item_features"], dtype=torch.float32, device=self.device).unsqueeze(0)
            page_features = torch.tensor(online["page_features"], dtype=torch.float32, device=self.device).unsqueeze(0)
            mask = torch.ones((1, item_features.shape[1]), dtype=torch.bool, device=self.device)
            with torch.no_grad():
                shares = self.online_slate_allocator_head.predict_shares(item_features, page_features, mask=mask)
            alloc_bonus = shares.squeeze(0).detach().cpu().numpy().astype(np.float32)

        combined = base_norm + float(self.online_slate_allocator_scale) * alloc_bonus
        order = np.argsort(-combined).tolist()
        if self.slate_value_head is None or float(self.slate_value_scale) == 0.0:
            return [int(candidate_pool[idx]) for idx in order[: int(self.slate_size)]]
        if bool(self.slate_full_rerank):
            candidate_slates = self._build_full_candidate_slates(order)
            best_positions = None
            best_value = None
            for slate_positions in candidate_slates:
                slate_candidate_idxs = [int(candidate_pool[p]) for p in slate_positions]
                slate_base_scores = [float(base_scores[p]) for p in slate_positions]
                value = self._score_slate_value(history_items, cand_iids, slate_candidate_idxs, slate_base_scores)
                value = float(self.slate_value_scale) * value + 1e-4 * float(sum(combined[p] for p in slate_positions))
                if best_value is None or value > best_value:
                    best_value = value
                    best_positions = [int(p) for p in slate_positions]
            if best_positions:
                return [int(candidate_pool[pos]) for pos in best_positions[: int(self.slate_size)]]

        greedy_limit = min(len(order), max(int(self.slate_greedy_candidates), int(self.slate_size)))
        selected_positions: List[int] = []
        remaining_positions: List[int] = list(order)
        while len(selected_positions) < int(self.slate_size) and remaining_positions:
            trial_positions = remaining_positions[:greedy_limit]
            best_pos = None
            best_value = None
            for pos in trial_positions:
                if pos in selected_positions:
                    continue
                tentative = selected_positions + [pos]
                fill = [x for x in order if x not in tentative]
                slate_positions = tentative + fill[: max(0, int(self.slate_size) - len(tentative))]
                slate_candidate_idxs = [int(candidate_pool[p]) for p in slate_positions]
                slate_base_scores = [float(base_scores[p]) for p in slate_positions]
                value = self._score_slate_value(history_items, cand_iids, slate_candidate_idxs, slate_base_scores)
                value = float(self.slate_value_scale) * value + 1e-4 * float(combined[pos])
                if best_value is None or value > best_value:
                    best_value = value
                    best_pos = int(pos)
            if best_pos is None:
                break
            selected_positions.append(int(best_pos))
            remaining_positions = [x for x in remaining_positions if int(x) != int(best_pos)]

        if len(selected_positions) < int(self.slate_size):
            for pos in order:
                if pos not in selected_positions:
                    selected_positions.append(int(pos))
                if len(selected_positions) >= int(self.slate_size):
                    break
        return [int(candidate_pool[pos]) for pos in selected_positions[: int(self.slate_size)]]

    def _build_full_candidate_slates(self, order: Sequence[int]) -> List[List[int]]:
        pool = [int(x) for x in order]
        if len(pool) < int(self.slate_size):
            return []
        slates: List[List[int]] = []
        top = pool[: int(self.slate_size)]
        slates.append([int(x) for x in top])

        for start in range(1, max(1, len(pool) - int(self.slate_size) + 1)):
            slate = pool[start:start + int(self.slate_size)]
            if len(slate) == int(self.slate_size):
                slates.append([int(x) for x in slate])
            if len(slates) >= int(self.slate_num_candidate_slates):
                break

        if len(slates) < int(self.slate_num_candidate_slates):
            top_slate = list(top)
            for repl_idx in range(int(self.slate_size)):
                for pool_idx in range(int(self.slate_size), len(pool)):
                    slate = list(top_slate)
                    slate[repl_idx] = int(pool[pool_idx])
                    if len(set(slate)) != int(self.slate_size):
                        continue
                    slates.append([int(x) for x in slate])
                    if len(slates) >= int(self.slate_num_candidate_slates):
                        break
                if len(slates) >= int(self.slate_num_candidate_slates):
                    break

        uniq: List[List[int]] = []
        seen = set()
        for slate in slates:
            key = tuple(int(x) for x in slate)
            if key in seen:
                continue
            seen.add(key)
            uniq.append([int(x) for x in slate])
            if len(uniq) >= int(self.slate_num_candidate_slates):
                break
        return uniq


def parse_args():
    parser = argparse.ArgumentParser()
    parser = KREnvironment_WholeSession_GPU.parse_model_args(parser)
    parser.add_argument("--tiger_ckpt", type=str, required=True)
    parser.add_argument("--sid_mapping_path", type=str, required=True)
    parser.add_argument("--phase2_head_path", type=str, default="")
    parser.add_argument("--phase2_meta_path", type=str, default="")
    parser.add_argument("--phase2_blend_scale", type=float, default=0.20)
    parser.add_argument("--phase2_decode_topk", type=int, default=0, help="0 means use full beam width.")
    parser.add_argument("--phase4_prefix_head_path", type=str, default="")
    parser.add_argument("--phase4_prefix_meta_path", type=str, default="")
    parser.add_argument("--phase4_prefix_scale", type=float, default=0.0)
    parser.add_argument("--phase3_actor_head_path", type=str, default="")
    parser.add_argument("--phase3_actor_meta_path", type=str, default="")
    parser.add_argument("--phase3_actor_scale", type=float, default=0.0)
    parser.add_argument("--phase5_token_actor_head_path", type=str, default="")
    parser.add_argument("--phase5_token_actor_meta_path", type=str, default="")
    parser.add_argument("--phase5_token_actor_scale", type=float, default=0.0)
    parser.add_argument("--phase6_joint_head_path", type=str, default="")
    parser.add_argument("--phase6_joint_meta_path", type=str, default="")
    parser.add_argument("--phase6_prefix_scale", type=float, default=0.0)
    parser.add_argument("--phase6_token_actor_scale", type=float, default=0.0)
    parser.add_argument("--online_slate_allocator_head_path", type=str, default="")
    parser.add_argument("--online_slate_allocator_meta_path", type=str, default="")
    parser.add_argument("--online_slate_allocator_scale", type=float, default=0.0)
    parser.add_argument("--slate_value_head_path", type=str, default="")
    parser.add_argument("--slate_value_meta_path", type=str, default="")
    parser.add_argument("--slate_value_scale", type=float, default=0.0)
    parser.add_argument("--slate_rerank_pool", type=int, default=12)
    parser.add_argument("--slate_greedy_candidates", type=int, default=8)
    parser.add_argument("--slate_full_rerank", action="store_true")
    parser.add_argument("--slate_num_candidate_slates", type=int, default=6)
    parser.add_argument("--fast_base_generate", action="store_true", help="When no Phase2 head is loaded, use original TIGER generate() path for fast base decoding.")
    parser.add_argument("--disable_decoder_kv_cache", action="store_true", help="Disable decoder KV-cache in custom beam decode and fall back to full-prefix decode.")
    parser.add_argument("--trace_path", type=str, default="", help="Optional JSONL output for complete episode traces.")

    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--max_steps_per_episode", type=int, default=20)
    parser.add_argument("--eval_log_path", type=str, default="../output/KuaiRand_Pure/eval/tiger_phase2_blend_eval.log")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--d_kv", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--feed_forward_proj", type=str, default="relu")

    parser.add_argument("--max_hist_items", type=int, default=50)
    parser.add_argument("--beam_width", type=int, default=50)
    parser.add_argument("--random_topk_sample", type=int, default=0, help="If >0, uniformly sample the final slate from the top-k ranked candidates.")
    parser.add_argument("--random_item_prob", type=float, default=0.0, help="Per-page probability of overriding beam selection with a random valid candidate item.")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    size_cfg = infer_model_size_args(args.model_size)
    args.num_layers = int(size_cfg["num_layers"])
    args.num_decoder_layers = int(size_cfg["num_decoder_layers"])
    args.d_model = int(size_cfg["d_model"])
    args.d_ff = int(size_cfg["d_ff"])
    args.num_heads = int(size_cfg["num_heads"])
    args.d_kv = int(size_cfg["d_kv"])
    return args


def run_eval(args):
    requested_batch_size = int(args.episode_batch_size)
    effective_batch_size = min(requested_batch_size, int(args.num_episodes))
    if effective_batch_size <= 0:
        raise ValueError("num_episodes must be positive.")
    if effective_batch_size != requested_batch_size:
        print(
            f"[EvalAdjust] episode_batch_size={requested_batch_size} > num_episodes={int(args.num_episodes)}; "
            f"use batch_size={effective_batch_size} to avoid early-finish bias."
        )
        args.episode_batch_size = int(effective_batch_size)
    device = torch.device(args.device)
    print(f"[Info] Using device: {device}")
    env = KREnvironment_WholeSession_GPU(args)
    model, sid_depth, codebook_size = load_tiger_model(
        tiger_ckpt=str(args.tiger_ckpt),
        sid_mapping_path=str(args.sid_mapping_path),
        num_layers=int(args.num_layers),
        num_decoder_layers=int(args.num_decoder_layers),
        d_model=int(args.d_model),
        d_ff=int(args.d_ff),
        num_heads=int(args.num_heads),
        d_kv=int(args.d_kv),
        dropout_rate=float(args.dropout_rate),
        feed_forward_proj=str(args.feed_forward_proj),
        device=device,
    )
    iid2sid_tok, sid2iid_map = build_iid2sid_tokens(env.reader, args.sid_mapping_path, sid_depth, device)

    phase2_head = None
    phase4_prefix_head = None
    phase3_actor_head = None
    phase5_token_actor_head = None
    phase6_plan_head = None
    phase6_prefix_head = None
    phase6_token_actor_head = None
    online_slate_allocator_head = None
    slate_value_head = None
    if args.phase2_head_path:
        meta_path = str(args.phase2_meta_path).strip()
        if not meta_path:
            meta_guess = Path(args.phase2_head_path).with_name("phase2_blend_meta.json")
            meta_path = str(meta_guess)
        phase2_head, phase2_meta = load_phase2_head(args.phase2_head_path, meta_path, device)
        print(f"[Phase2] Loaded head: {args.phase2_head_path}")
        print(f"[Phase2] Meta: sid_depth={phase2_meta.get('sid_depth')} vocab={phase2_meta.get('vocab_size')}")
    if args.phase4_prefix_head_path:
        prefix_meta_path = str(args.phase4_prefix_meta_path).strip()
        if not prefix_meta_path:
            prefix_meta_guess = Path(args.phase4_prefix_head_path).with_name("prefix_critic_meta.json")
            prefix_meta_path = str(prefix_meta_guess)
        phase4_prefix_head, prefix_meta = load_prefix_value_head(args.phase4_prefix_head_path, prefix_meta_path, device)
        print(f"[Phase4] Loaded prefix head: {args.phase4_prefix_head_path}")
        print(f"[Phase4] Meta: sid_depth={prefix_meta.get('sid_depth')} vocab={prefix_meta.get('vocab_size')}")
    if args.phase3_actor_head_path:
        actor_meta_path = str(args.phase3_actor_meta_path).strip()
        if not actor_meta_path:
            actor_meta_guess = Path(args.phase3_actor_head_path).with_name("phase3_actor_meta.json")
            actor_meta_path = str(actor_meta_guess)
        phase3_actor_head, phase3_meta = load_phase3_actor_head(args.phase3_actor_head_path, actor_meta_path, device)
        print(f"[Phase3] Loaded actor head: {args.phase3_actor_head_path}")
        print(f"[Phase3] Meta: sid_depth={phase3_meta.get('sid_depth')} vocab={phase3_meta.get('vocab_size')}")
    if args.phase5_token_actor_head_path:
        phase5_meta_path = str(args.phase5_token_actor_meta_path).strip()
        if not phase5_meta_path:
            phase5_meta_guess = Path(args.phase5_token_actor_head_path).with_name("phase5_token_actor_meta.json")
            phase5_meta_path = str(phase5_meta_guess)
        phase5_token_actor_head, phase5_meta = load_phase5_token_actor_head(args.phase5_token_actor_head_path, phase5_meta_path, device)
        print(f"[Phase5] Loaded token actor head: {args.phase5_token_actor_head_path}")
        print(f"[Phase5] Meta: sid_depth={phase5_meta.get('sid_depth')} vocab={phase5_meta.get('vocab_size')}")
    if args.phase6_joint_head_path:
        phase6_meta_path = str(args.phase6_joint_meta_path).strip()
        if not phase6_meta_path:
            phase6_meta_guess = Path(args.phase6_joint_head_path).with_name("phase6_joint_meta.json")
            phase6_meta_path = str(phase6_meta_guess)
        phase6_bundle, phase6_meta = load_phase6_joint_bundle(args.phase6_joint_head_path, phase6_meta_path, device)
        phase6_plan_head = phase6_bundle["plan_head"]
        phase6_prefix_head = phase6_bundle["prefix_head"]
        phase6_token_actor_head = phase6_bundle["token_actor_head"]
        print(f"[Phase6] Loaded joint heads: {args.phase6_joint_head_path}")
        print(f"[Phase6] Meta: plans={phase6_meta.get('plan_names')} vocab={phase6_meta.get('vocab_size')}")
    if args.online_slate_allocator_head_path:
        allocator_meta_path = str(args.online_slate_allocator_meta_path).strip()
        if not allocator_meta_path:
            allocator_meta_guess = Path(args.online_slate_allocator_head_path).with_name("online_slate_allocator_meta.json")
            allocator_meta_path = str(allocator_meta_guess)
        online_slate_allocator_head, allocator_meta = load_online_slate_allocator(args.online_slate_allocator_head_path, allocator_meta_path, device)
        print(f"[SlateOnline] Loaded allocator head: {args.online_slate_allocator_head_path}")
        print(f"[SlateOnline] Meta: item_dim={allocator_meta.get('item_dim')} page_dim={allocator_meta.get('page_dim')}")
    if args.slate_value_head_path:
        slate_value_meta_path = str(args.slate_value_meta_path).strip()
        if not slate_value_meta_path:
            slate_value_meta_guess = Path(args.slate_value_head_path).with_name("slate_value_meta.json")
            slate_value_meta_path = str(slate_value_meta_guess)
        slate_value_head, slate_value_meta = load_slate_value_head(args.slate_value_head_path, slate_value_meta_path, device)
        print(f"[SlateCritic] Loaded value head: {args.slate_value_head_path}")
        print(f"[SlateCritic] Meta: item_dim={slate_value_meta.get('item_dim')} page_dim={slate_value_meta.get('page_dim')}")

    policy = TigerPhase2BlendPolicy(
        model=model,
        iid2sid_tok=iid2sid_tok,
        sid2iid_map_tok=sid2iid_map,
        sid_depth=sid_depth,
        device=device,
        slate_size=args.slate_size,
        reader=env.reader,
        args=args,
        phase2_head=phase2_head,
        phase4_prefix_head=phase4_prefix_head,
        phase3_actor_head=phase3_actor_head,
        phase5_token_actor_head=phase5_token_actor_head,
        phase6_plan_head=phase6_plan_head,
        phase6_prefix_head=phase6_prefix_head,
        phase6_token_actor_head=phase6_token_actor_head,
        online_slate_allocator_head=online_slate_allocator_head,
        slate_value_head=slate_value_head,
    )
    print(
        f"[Eval] TIGER Phase2 Blend | Beam Width={args.beam_width} | "
        f"BlendScale={args.phase2_blend_scale:.3f} | PrefixScale={args.phase4_prefix_scale:.3f} | "
        f"ActorScale={args.phase3_actor_scale:.3f} | "
        f"TokenActorScale={args.phase5_token_actor_scale:.3f} | "
        f"JointPrefixScale={args.phase6_prefix_scale:.3f} | "
        f"JointTokenScale={args.phase6_token_actor_scale:.3f} | "
        f"SlateAllocScale={args.online_slate_allocator_scale:.3f} | "
        f"SlateValueScale={args.slate_value_scale:.3f} | "
        f"Slate Size={args.slate_size} | "
        f"DecKVCache={'off' if args.disable_decoder_kv_cache else 'on'}"
    )
    print(f"[Eval] SID depth={sid_depth}, codebook_size={codebook_size}")

    observation = env.reset({"batch_size": effective_batch_size})
    bsz = int(effective_batch_size)
    cur_returns = torch.zeros(bsz, device=device)
    cur_lengths = torch.zeros(bsz, device=device)
    finished = 0
    all_ret, all_len = [], []

    beh_names = None
    if hasattr(env, "response_types"):
        try:
            beh_names = list(env.response_types)
        except Exception:
            beh_names = None
    if beh_names is None:
        beh_names = ["is_click", "long_view", "is_like", "is_comment", "is_forward", "is_follow", "is_hate"]
    k_fb = len(beh_names)
    cur_beh_counts = torch.zeros(bsz, k_fb, device=device)
    cur_impr = torch.zeros(bsz, device=device)
    total_beh_counts = torch.zeros(k_fb, device=device)
    total_impr = 0.0
    response_weights = env.response_weights

    episode_ids = np.arange(bsz, dtype=np.int64)
    next_episode_id = int(bsz)
    page_indices = np.ones(bsz, dtype=np.int64)
    trace_buffers: Dict[int, List[Dict[str, Any]]] = {int(i): [] for i in episode_ids.tolist()}
    trace_fp = None
    if args.trace_path:
        trace_path = Path(args.trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_fp = trace_path.open("w", encoding="utf-8")

    try:
        while finished < args.num_episodes:
            try:
                cand = env.get_candidate_info(feed_dict=None)
            except Exception:
                cand = env.get_candidate_info()

            hist_snapshot = observation["user_history"]["history"].detach().cpu().clone()
            uid_snapshot = observation["user_profile"]["user_id"].detach().cpu().clone()
            cand_iids = cand["item_id"].detach().cpu().numpy()
            action = policy.act(observation, cand)
            next_obs, resp, _ = env.step({"action": action})

            im = resp.get("immediate_response", None)
            if im is None:
                raise RuntimeError("Missing immediate_response in environment output.")
            if isinstance(im, np.ndarray):
                im = torch.from_numpy(im)
            im = im.to(device).float()
            cur_impr += float(im.size(1))
            k_eff = min(k_fb, int(im.size(2)))
            if k_eff < k_fb:
                cur_beh_counts[:, :k_eff] += im[:, :, :k_eff].sum(dim=1)
            else:
                cur_beh_counts += im.sum(dim=1)

            rw_dict = {
                "immediate_response": resp["immediate_response"],
                "immediate_response_weight": response_weights,
            }
            step_r = get_immediate_reward(rw_dict).to(device)
            cur_returns += step_r
            cur_lengths += 1

            done = resp["done"]
            if isinstance(done, np.ndarray):
                done = torch.from_numpy(done)
            done = done.to(device).bool()
            action_cpu = action.detach().cpu().numpy()
            im_cpu = im.detach().cpu().numpy()
            step_r_cpu = step_r.detach().cpu().numpy()
            if isinstance(response_weights, torch.Tensor):
                response_weights_cpu = response_weights.detach().cpu().numpy().astype(np.float32)
            else:
                response_weights_cpu = np.asarray(response_weights, dtype=np.float32)

            for row_idx in range(bsz):
                selected_local = action_cpu[row_idx].tolist()
                selected_global = [int(cand_iids[int(idx)]) for idx in selected_local if 0 <= int(idx) < len(cand_iids)]
                first_iid = int(selected_global[0]) if selected_global else -1
                action_debug = {}
                if hasattr(policy, "last_action_debug") and row_idx < len(policy.last_action_debug):
                    action_debug = dict(policy.last_action_debug[row_idx])
                sid_tokens = []
                if 0 <= first_iid < int(iid2sid_tok.shape[0]):
                    sid_tokens = [int(x) for x in iid2sid_tok[first_iid].detach().cpu().tolist()]
                selected_sid_tokens_list: List[List[int]] = []
                for gid in selected_global:
                    if 0 <= int(gid) < int(iid2sid_tok.shape[0]):
                        selected_sid_tokens_list.append([int(x) for x in iid2sid_tok[int(gid)].detach().cpu().tolist()])
                    else:
                        selected_sid_tokens_list.append([])
                selected_responses = im_cpu[row_idx].astype(float).tolist()
                if im_cpu.shape[2] == int(response_weights_cpu.shape[0]):
                    per_item_reward = (im_cpu[row_idx] * response_weights_cpu.reshape(1, -1)).sum(axis=1).astype(float).tolist()
                else:
                    per_item_reward = im_cpu[row_idx].sum(axis=1).astype(float).tolist()
                record = {
                    "episode_id": int(episode_ids[row_idx]),
                    "page_index": int(page_indices[row_idx]),
                    "user_id": int(uid_snapshot[row_idx].item()),
                    "history_items": [int(x) for x in hist_snapshot[row_idx].tolist() if int(x) > 0],
                    "selected_item_id": int(first_iid),
                    "selected_item_ids": [int(x) for x in selected_global],
                    "selected_sid_tokens": sid_tokens,
                    "selected_sid_tokens_list": selected_sid_tokens_list,
                    "selected_response": im_cpu[row_idx, 0, :].astype(float).tolist() if im_cpu.shape[1] > 0 else [],
                    "selected_responses": selected_responses,
                    "selected_item_rewards": per_item_reward,
                    "response_weights": response_weights_cpu.astype(float).tolist(),
                    "step_reward": float(step_r_cpu[row_idx]),
                    "done": bool(done[row_idx].item()),
                    "selection_mode": str(action_debug.get("selection_mode", "greedy")),
                    "selection_topk": int(action_debug.get("selection_topk", 0)),
                    "selection_random_item_prob": float(action_debug.get("selection_random_item_prob", 0.0)),
                    "selection_sampled_ranks": [int(x) for x in action_debug.get("selection_sampled_ranks", [])],
                    "selection_greedy_local": [int(x) for x in action_debug.get("selection_greedy_local", [])],
                    "selection_sampled_local_pre_fill": [int(x) for x in action_debug.get("selection_sampled_local_pre_fill", [])],
                    "selection_final_local": [int(x) for x in action_debug.get("selection_final_local", [])],
                }
                trace_buffers.setdefault(int(episode_ids[row_idx]), []).append(record)

            if done.any():
                idxs = torch.nonzero(done, as_tuple=False).squeeze(-1)
                for idx in idxs.tolist():
                    episode_id = int(episode_ids[idx])
                    if finished < args.num_episodes:
                        all_ret.append(float(cur_returns[idx].item()))
                        all_len.append(float(cur_lengths[idx].item()))
                        finished += 1
                        total_beh_counts += cur_beh_counts[idx]
                        total_impr += float(cur_impr[idx].item())
                        if trace_fp is not None:
                            for rec in trace_buffers.get(episode_id, []):
                                trace_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        if finished % args.log_every == 0:
                            print(
                                f"Progress {finished}/{args.num_episodes} | "
                                f"Ret: {np.mean(all_ret):.4f} | Len: {np.mean(all_len):.2f}"
                            )
                    trace_buffers.pop(episode_id, None)
                    episode_ids[idx] = next_episode_id
                    next_episode_id += 1
                    page_indices[idx] = 1

                cur_returns[done] = 0
                cur_lengths[done] = 0
                cur_beh_counts[done] = 0
                cur_impr[done] = 0

            keep_mask = (~done).detach().cpu().numpy()
            page_indices[keep_mask] += 1
            observation = next_obs
    finally:
        if trace_fp is not None:
            trace_fp.close()

    if not all_ret:
        print("[WARN] no completed episodes.")
        return

    total_ret = float(np.mean(all_ret))
    depth = float(np.mean(all_len))
    avg_step_r = total_ret / depth if depth > 0 else 0.0
    coverage = 0.0
    ild = 0.0
    if hasattr(env, "get_report"):
        env_report = env.get_report()
        coverage = float(env_report.get("coverage", 0.0))
        ild = float(env_report.get("ILD", 0.0))

    print("=" * 40)
    print(f"Total Reward: {total_ret:.4f}")
    print(f"Depth: {depth:.2f}")
    print(f"Avg Step Reward: {avg_step_r:.4f}")
    print(f"Coverage: {coverage:.2f}")
    print(f"ILD: {ild:.4f}")
    print("Table-style metrics:")
    print(f"Depth: {depth:.2f}")
    print(f"Average reward: {avg_step_r:.4f}")
    print(f"Total reward: {total_ret:.4f}")
    print(f"Coverage: {coverage:.2f}")
    print(f"ILD: {ild:.4f}")
    print("Behavior rates (count / impressions):")
    if total_impr <= 0:
        print("  [WARN] total_impr=0")
    else:
        for k, name in enumerate(beh_names):
            cnt = float(total_beh_counts[k].item())
            rate = 100.0 * cnt / total_impr
            print(f"  {name}: {int(round(cnt))}/{int(round(total_impr))} ({rate:.4f}%)")
    print("=" * 40)


if __name__ == "__main__":
    args = parse_args()
    utils.set_random_seed(args.seed)
    run_eval(args)
