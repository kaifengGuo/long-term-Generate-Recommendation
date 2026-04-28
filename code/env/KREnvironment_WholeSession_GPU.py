import numpy as np
import torch
import random
import json
from copy import deepcopy
from argparse import Namespace
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import torch.nn.functional as F

import utils
from reader import *
from model.simulator import *
from model.simulator.KRUserExitBernoulli import KRUserExitBernoulli
from env.BaseRLEnvironment import BaseRLEnvironment

def neg_sample_calibrate_probs(p, s, eps=1e-8):
    """
    p:  (..., n_feedback)  in [0,1]  (URM predicted under neg down-sampling / neg re-weight)
    s:  (..., n_feedback)  keep probability for negatives (or pos/neg ratio), typically small for rare events
    return: p_real in [0,1]
    
    Formula:
      odds_real = s * odds_model
      p_real = (s*p) / ((1-p) + s*p)
    """
    p = p.clamp(eps, 1.0 - eps)
    s = s.clamp(min=eps)
    return (s * p) / ((1.0 - p) + s * p)


class KREnvironment_WholeSession_GPU(BaseRLEnvironment):
    '''
    KuaiRand simulated environment for consecutive list-wise recommendation
    Main interface:
    - parse_model_args: for hyperparameters
    - reset: reset online environment, monitor, and corresponding initial observation
    - step: action --> new observation, user feedbacks, and other updated information
    - get_candidate_info: obtain the entire item candidate pool
    Main Components:
    - data reader: self.reader for user profile&history sampler
    - user immediate response model: see self.get_response
    - no user leave model: see self.get_leave_signal
    - candidate item pool: self.candidate_ids, self.candidate_item_meta
    - history monitor: self.env_history, not set up until self.reset
    '''

    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - uirm_log_path
        - slate_size
        - episode_batch_size
        - item_correlation
        - single_response
        - from BaseRLEnvironment
            - max_step_per_episode
            - initial_temper
        '''
        parser = BaseRLEnvironment.parse_model_args(parser)
        parser.add_argument('--uirm_log_path', type=str, required=True,
                            help='log path for pretrained user immediate response model')
        parser.add_argument('--slate_size', type=int, required=6,
                            help='number of item per recommendation slate')
        parser.add_argument('--episode_batch_size', type=int, default=32,
                            help='episode sample batch size')
        parser.add_argument('--item_correlation', type=float, default=0,
                            help='magnitude of item correlation')
        parser.add_argument('--single_response', action='store_true',
                            help='only include the first feedback as reward signal')
        parser.add_argument('--temper_consume_mode', type=str, default='position_weighted',
                            choices=['legacy_mean', 'position_weighted'],
                            help='How slate feedback is aggregated into temper consumption.')
        parser.add_argument('--temper_consume_decay', type=float, default=0.7,
                            help='Geometric position decay for temper consumption. Position j uses decay^j.')
        parser.add_argument('--temper_cost_offset', type=float, default=2.0,
                            help='Baseline cost subtracted from the aggregated temper reward each step.')
        parser.add_argument('--temper_max_drop', type=float, default=2.0,
                            help='Maximum temper drop magnitude allowed per step.')
        parser.add_argument('--history_consume_mode', type=str, default='expected_prefix',
                            choices=['full_slate', 'expected_prefix'],
                            help='How many top-ranked items are appended into user history each step.')
        parser.add_argument('--history_consume_decay', type=float, default=-1.0,
                            help='Optional decay override for history consumption. Negative uses temper_consume_decay.')
        parser.add_argument('--use_leave_model', action='store_true',
                            help='Enable learned leave model. If disabled, use temper-based leave.')
        parser.add_argument('--leave_model_path', type=str, default='',
                            help='Optional checkpoint path of learned leave-probability model.')
        parser.add_argument('--leave_prob_clip', type=float, default=0.999,
                            help='Upper clip for leave probability before Bernoulli sampling.')
        parser.add_argument('--leave_logit_temperature', type=float, default=1.0,
                            help='Temperature on leave logits. >1.0 lowers extreme probabilities.')
        parser.add_argument('--leave_logit_bias', type=float, default=0.0,
                            help='Additive bias on leave logits. Negative value reduces leave probability.')
        parser.add_argument('--leave_min_survival_steps', type=int, default=0,
                            help='Force no leave for initial steps [0, leave_min_survival_steps).')
        parser.add_argument('--leave_early_step_max', type=int, default=5,
                            help='Max early step index for probability scaling ramp.')
        parser.add_argument('--leave_early_prob_scale', type=float, default=1.0,
                            help='Scale factor at step=0, linearly ramping to 1.0 by leave_early_step_max.')
        parser.add_argument('--leave_debug_log_path', type=str, default='',
                            help='Optional jsonl path to dump per-step leave probability cases.')
        parser.add_argument('--leave_debug_max_records', type=int, default=0,
                            help='Max records to dump for leave probability debug. 0 disables.')
        return parser

    def __init__(self, args):
        '''
        from BaseRLEnvironment:
            self.max_step_per_episode
            self.initial_temper
        self.uirm_log_path
        self.slate_size
        self.rho
        self.immediate_response_stats: reader statistics for user response model
        self.immediate_response_model: the ground truth user response model
        self.max_hist_len
        self.response_types
        self.response_dim: number of feedback_type
        self.response_weights
        self.reader
        self.candidate_iids: [encoded item id]
        self.candidate_item_meta: {'if_{feature_name}': (n_item, feature_dim)}
        self.n_candidate
        self.candidate_item_encoding: (n_item, item_enc_dim)
        self.gt_state_dim: ground truth user state vector dimension
        self.action_dim: slate size
        self.observation_space: see reader.get_statistics()
        self.action_space: n_condidate
        '''
        super(KREnvironment_WholeSession_GPU, self).__init__(args)
        self.uirm_log_path = args.uirm_log_path
        self.slate_size = args.slate_size
        self.episode_batch_size = args.episode_batch_size
        self.rho = args.item_correlation
        self.single_response = args.single_response
        self.temper_consume_mode = str(getattr(args, "temper_consume_mode", "position_weighted"))
        self.temper_consume_decay = float(getattr(args, "temper_consume_decay", 0.7))
        self.temper_cost_offset = float(getattr(args, "temper_cost_offset", 2.0))
        self.temper_max_drop = float(getattr(args, "temper_max_drop", 2.0))
        self.history_consume_mode = str(getattr(args, "history_consume_mode", "expected_prefix"))
        self.history_consume_decay = float(getattr(args, "history_consume_decay", -1.0))
        self.use_leave_model = bool(getattr(args, "use_leave_model", False))
        self.leave_model_path = getattr(args, "leave_model_path", "")
        self.leave_prob_clip = float(getattr(args, "leave_prob_clip", 0.999))
        self.leave_logit_temperature = float(getattr(args, "leave_logit_temperature", 1.0))
        self.leave_logit_bias = float(getattr(args, "leave_logit_bias", 0.0))
        self.leave_min_survival_steps = int(getattr(args, "leave_min_survival_steps", 0))
        self.leave_early_step_max = int(getattr(args, "leave_early_step_max", 5))
        self.leave_early_prob_scale = float(getattr(args, "leave_early_prob_scale", 1.0))
        self.leave_debug_log_path = getattr(args, "leave_debug_log_path", "")
        self.leave_debug_max_records = int(getattr(args, "leave_debug_max_records", 0))
        self.leave_model = None
        self.leave_feature_spec = None
        self._leave_debug_step = 0
        self._leave_debug_records_written = 0
        self._position_weight_cache = {}
        self._last_history_consume_count = 0


        _m1 = getattr(args, "max_step_per_episode", None)
        _m2 = getattr(args, "max_steps_per_episode", None)
        _m = _m1 if _m1 is not None else _m2
        if _m is not None:
            try:
                self.max_step_per_episode = int(_m)
            except Exception:
                pass
        if hasattr(self, "max_step_per_episode"):
            self.max_steps_per_episode = int(getattr(self, "max_step_per_episode", 0))
        infile = open(args.uirm_log_path, 'r')
        class_args = eval(infile.readline())  # example: Namespace(model='RL4RSUserResponse', reader='RL4RSDataReader')
        model_args = eval(infile.readline())  # model parameters in Namespace
        print("Environment arguments: \n" + str(model_args))
        infile.close()
        print("Loading raw data")
        assert (
                           class_args.reader == 'KRMBSeqReader' or class_args.reader == 'MLSeqReader') and 'KRMBUserResponse' in class_args.model

        print("Load user sequence reader")
        reader, reader_args = self.get_reader(args.uirm_log_path)  # definition in base
        self.reader = reader
        print(self.reader.get_statistics())

        print("Load immediate user response model")
        uirm_stats, uirm_model, uirm_args = self.get_user_model(args.uirm_log_path, args.device)  # definition in base
        self.immediate_response_stats = uirm_stats
        self.immediate_response_model = uirm_model
        self.max_hist_len = uirm_stats['max_seq_len']
        self.response_types = uirm_stats['feedback_type']
        self.response_dim = len(self.response_types)
        self.response_weights = torch.tensor(list(self.reader.get_response_weights().values())).to(torch.float).to(
            args.device)
        if args.single_response:
            self.response_weights = torch.zeros_like(self.response_weights)
            self.response_weights[0] = 1

        expected_prefix = self._get_history_consume_length(self.slate_size)
        print(
            "[TemperMode] "
            f"mode={self.temper_consume_mode}, "
            f"decay={self.temper_consume_decay:.3f}, "
            f"cost={self.temper_cost_offset:.3f}, "
            f"max_drop={self.temper_max_drop:.3f}, "
            f"history_mode={self.history_consume_mode}, "
            f"expected_prefix={expected_prefix}/{self.slate_size}"
        )

        if self.use_leave_model:
            if not self.leave_model_path:
                raise ValueError("use_leave_model=True but leave_model_path is empty.")
            self.leave_model, self.leave_feature_spec = KRUserExitBernoulli.load_from_checkpoint(
                self.leave_model_path, args.device
            )
            print(f"[LeaveModel] loaded from: {self.leave_model_path}")
            print(f"[LeaveModel] input_dim={self.leave_feature_spec['input_dim']}")
            print(
                "[LeaveModel] calibration: "
                f"temp={self.leave_logit_temperature}, "
                f"bias={self.leave_logit_bias}, "
                f"min_survival_steps={self.leave_min_survival_steps}, "
                f"early_step_max={self.leave_early_step_max}, "
                f"early_prob_scale={self.leave_early_prob_scale}"
            )

        print("Setup candidate item pool")

        self.candidate_iids = torch.tensor([reader.item_id_vocab[iid] for iid in reader.items]).to(self.device)

        candidate_meta = [reader.get_item_meta_data(iid) for iid in reader.items]
        self.candidate_item_meta = {}
        self.n_candidate = len(candidate_meta)
        for k in candidate_meta[0]:
            self.candidate_item_meta[k] = torch.FloatTensor(np.concatenate([meta[k] for meta in candidate_meta]))\
                .view(self.n_candidate, -1).to(self.device)

        item_enc, _ = self.immediate_response_model.get_item_encoding(self.candidate_iids,
                                                                      {k[3:]: v for k, v in
                                                                       self.candidate_item_meta.items()}, 1)
        self.candidate_item_encoding = item_enc.view(-1, self.immediate_response_model.enc_dim)

        self.gt_state_dim = self.immediate_response_model.state_dim
        self.action_dim = self.slate_size
        self.observation_space = self.reader.get_statistics()
        self.action_space = self.n_candidate

        self.immediate_response_model.to(args.device)
        self.immediate_response_model.device = args.device

    def _build_position_examination_weights(self, slate_size, normalize=False, decay=None):
        slate_size = max(1, int(slate_size))
        use_decay = self.temper_consume_decay if decay is None else decay
        use_decay = float(min(max(use_decay, 1e-3), 1.0))
        cache_key = (slate_size, round(use_decay, 6), bool(normalize))
        cached = self._position_weight_cache.get(cache_key)
        if cached is not None:
            return cached

        if abs(use_decay - 1.0) < 1e-6:
            weights = torch.ones(slate_size, dtype=torch.float32, device=self.device)
        else:
            positions = torch.arange(slate_size, dtype=torch.float32, device=self.device)
            weights = torch.pow(
                torch.full((slate_size,), use_decay, dtype=torch.float32, device=self.device),
                positions,
            )
        if normalize:
            weights = weights / weights.sum().clamp(min=1e-6)
        self._position_weight_cache[cache_key] = weights
        return weights

    def _aggregate_temper_reward(self, response_dict):
        point_reward = response_dict['immediate_response'] * self.response_weights.view(1, 1, -1)
        combined_reward = torch.sum(point_reward, dim=2)
        if self.temper_consume_mode == 'legacy_mean':
            return torch.mean(combined_reward, dim=1)
        weights = self._build_position_examination_weights(combined_reward.shape[1], normalize=True)
        return torch.sum(combined_reward * weights.view(1, -1), dim=1)

    def _get_history_consume_length(self, slate_size):
        slate_size = max(1, int(slate_size))
        if self.history_consume_mode == 'full_slate':
            return slate_size
        decay = self.history_consume_decay
        if decay <= 0:
            decay = self.temper_consume_decay
        weights = self._build_position_examination_weights(slate_size, normalize=False, decay=decay)
        expected_mass = float(weights.sum().item())
        consume_len = int(np.floor(expected_mass + 0.5))
        return max(1, min(slate_size, consume_len))


    def get_candidate_info(self, feed_dict=None, all_item=True):
        '''
        Add entire item pool as candidate for the feed_dict
        @input:
        - all_item: whether obtain all item features from candidate pool
        - feed_dict
        @output:
        - candidate_info: {'item_id': (L,),
                           'if_{feature_name}': (n_item, feature_dim)}
        '''
        if all_item:
            candidate_info = {'item_id': self.candidate_iids}
            candidate_info.update(self.candidate_item_meta)
        else:
            if feed_dict is None:
                raise ValueError("feed_dict is required when all_item=False")

            candidate_info = {'item_id': feed_dict['item_id']}
            indices = feed_dict['item_id'] - 1
            candidate_info.update({k: v[indices] for k, v in self.candidate_item_meta.items()})
        return candidate_info

    def reset(self, params={'empty_history': True}):
        '''
        Reset environment with new sampled users
        @input:
        - params: {'batch_size': scalar,
                   'empty_history': True if start from empty history,
                   'initial_history': start with initial history}
        @process:
        - self.batch_iter
        - self.current_observation
        - self.current_step
        - self.current_temper
        - self.env_history
        @output:
        - observation: {'user_profile': {'user_id': (B,),
                                         'uf_{feature_name}': (B, feature_dim)},
                        'user_history': {'history': (B, max_H),
                                         'history_if_{feature_name}': (B, max_H, feature_dim),
                                         'history_{response}': (B, max_H),
                                         'history_length': (B, )}}
        '''
        if 'empty_history' not in params:
            params['empty_history'] = False

        if 'batch_size' in params:
            BS = params['batch_size']
        else:
            BS = self.episode_batch_size

        self.batch_iter = iter(DataLoader(self.reader, batch_size=BS, shuffle=True,
                                          pin_memory=True, num_workers=8))
        sample_info = next(self.batch_iter)
        self.sample_batch = self.get_observation_from_batch(sample_info)
        self.current_observation = self.sample_batch
        self.current_step = torch.zeros(self.episode_batch_size).to(self.device)
        self.current_sample_head_in_batch = BS

        self.current_temper = torch.ones(self.episode_batch_size).to(self.device) * self.initial_temper
        self.current_sum_reward = torch.zeros(self.episode_batch_size).to(self.device)

        self.env_history = {'step': [0.], 'leave': [], 'temper': [], 'consumed': [],
                            'coverage': [], 'ILD': []}

        return deepcopy(self.current_observation)

    def step(self, step_dict):
        '''
        users react to the recommendation action
        @input:
        - step_dict: {'action': (B, slate_size),
                      'action_features': (B, slate_size, item_dim) }
        @output:
        - new_observation: {'user_profile': {'user_id': (B,),
                                             'uf_{feature_name}': (B, feature_dim)},
                            'user_history': {'history': (B, max_H),
                                             'history_if_{feature_name}': (B, max_H, feature_dim),
                                             'history_{response}': (B, max_H),
                                             'history_length': (B, )}}
        - response_dict: {'immediate_response': see self.get_response@output - response_dict,
                          'done': (B,)}
        - update_info: see self.update_observation@output - update_info
        '''

        with torch.no_grad():
            action = step_dict['action']  # must be indices on candidate_ids

            response_dict = self.get_response(step_dict)
            response = response_dict['immediate_response']

            done_mask = self.get_leave_signal(None, None, response_dict)  # this will also change self.current_temper
            response_dict['done'] = done_mask

            update_info = self.update_observation(None, action, response, done_mask)

            self.current_step += 1
            max_step = getattr(self, "max_step_per_episode", None)
            if max_step is None:
                max_step = getattr(self, "max_steps_per_episode", None)
            try:
                max_step_i = int(max_step) if (max_step is not None) else None
            except Exception:
                max_step_i = None
            if (max_step_i is not None) and (max_step_i > 0):
                reach_limit = self.current_step >= max_step_i
                if reach_limit.any():
                    done_mask = done_mask | reach_limit
                    response_dict['done'] = done_mask
            n_leave = done_mask.sum()
            n_leave_int = int(n_leave.item())
            self.env_history['leave'].append(n_leave.item())
            self.env_history['temper'].append(torch.mean(self.current_temper).item())
            self.env_history['consumed'].append(float(self._last_history_consume_count))
            self.env_history['coverage'].append(response_dict['coverage'])
            self.env_history['ILD'].append(response_dict['ILD'])

            if n_leave_int > 0:
                final_steps = self.current_step[done_mask].detach().cpu().numpy()
                for fst in final_steps:
                    self.env_history['step'].append(fst)

                if self.current_sample_head_in_batch + n_leave_int < self.episode_batch_size:
                    head = self.current_sample_head_in_batch
                    tail = self.current_sample_head_in_batch + n_leave_int
                    for obs_key in ['user_profile', 'user_history']:
                        for k, v in self.sample_batch[obs_key].items():
                            dst = self.current_observation[obs_key][k]
                            src = v[head:tail]
                            if torch.is_tensor(dst) and torch.is_tensor(src):
                                if src.device != dst.device:
                                    src = src.to(dst.device)
                                if src.dtype != dst.dtype:
                                    src = src.to(dtype=dst.dtype)
                            dst[done_mask] = src
                    self.current_sample_head_in_batch += n_leave_int
                else:
                    sample_info = self.sample_new_batch_from_reader()
                    self.sample_batch = self.get_observation_from_batch(sample_info)
                    for obs_key in ['user_profile', 'user_history']:
                        for k, v in self.sample_batch[obs_key].items():
                            dst = self.current_observation[obs_key][k]
                            src = v[:n_leave_int]
                            if torch.is_tensor(dst) and torch.is_tensor(src):
                                if src.device != dst.device:
                                    src = src.to(dst.device)
                                if src.dtype != dst.dtype:
                                    src = src.to(dtype=dst.dtype)
                            dst[done_mask] = src
                    self.current_sample_head_in_batch = n_leave_int
                self.current_step[done_mask] *= 0
                self.current_temper[done_mask] *= 0
                self.current_temper[done_mask] += self.initial_temper
            else:
                self.env_history['step'].append(self.env_history['step'][-1])

        return deepcopy(self.current_observation), response_dict, update_info

    def get_response(self, step_dict):
        '''
        @input:
        - step_dict: {'action': (B, slate_size)}
        @output:
        - response_dict: {'immediate_response': (B, slate_size, n_feedback),
                          'user_state': (B, gt_state_dim),
                          'coverage': scalar,
                          'ILD': scalar}
        '''

        
        
        
        
        
    
    
        action = step_dict["action"]
        coverage = len(torch.unique(action))
        B = action.shape[0]
        slate = action.shape[1] if action.dim() == 2 else 1  # :  self.slate_size
    
        batch = {"item_id": self.candidate_iids[action]}
        batch.update(self.current_observation["user_profile"])
        batch.update(self.current_observation["user_history"])
        batch.update({k: v[action] for k, v in self.candidate_item_meta.items()})
    
        out_dict = self.immediate_response_model(batch)
    
        item_enc = self.candidate_item_encoding[action].view(B, slate, -1)
        item_enc_norm = F.normalize(item_enc, p=2.0, dim=-1)
        corr_factor = self.get_intra_slate_similarity(item_enc_norm)  # (B, slate)
    
        if "probs" in out_dict:
            p = out_dict["probs"]
        elif "preds" in out_dict:
            p = torch.sigmoid(out_dict["preds"])
        else:
            raise KeyError(f"URM output has no preds/probs. keys={list(out_dict.keys())}")
    
        if getattr(self, "rho", 0.0) != 0:
            p = p - corr_factor.view(B, slate, 1) * self.rho
    
        p = p.clamp(0.0, 1.0)
        if getattr(self, "enable_neg_sample_calibration", True):
            if (not hasattr(self, "_neg_keep_vec")) or (self._neg_keep_vec.device != p.device) or (self._neg_keep_vec.dtype != p.dtype):
                rates = getattr(self.reader, "response_neg_sample_rate", None)
                if rates is None:
                    rates = self.reader.get_response_weights()
        
                keep = []
                for name in self.response_types:
                    v = float(rates.get(name, 1.0))
                    keep.append(abs(v))  # hate 
                self._neg_keep_vec = torch.tensor(keep, device=p.device, dtype=p.dtype).view(1, 1, -1)
        
            s = self._neg_keep_vec  # (1,1,n_feedback)
            eps = 1e-8
            p = p.clamp(eps, 1.0 - eps)
            s = s.clamp(min=eps)
            p = (s * p) / ((1.0 - p) + s * p)
            p = p.clamp(0.0, 1.0)

        feedback_types = None
        if hasattr(self.immediate_response_model, "feedback_types"):
            feedback_types = list(self.immediate_response_model.feedback_types)
        elif hasattr(self, "response_types"):
            feedback_types = list(self.response_types)
        else:
            try:
                feedback_types = list(self.reader.get_statistics()["feedback_type"])
            except Exception:
                feedback_types = None
    
        if feedback_types is None:
            feedback_types = ["is_click", "long_view", "is_like", "is_comment", "is_forward", "is_follow", "is_hate"]
    
        idx = {name: i for i, name in enumerate(feedback_types)}
    
        if "is_click" not in idx:
            raise KeyError(f"feedback_types has no is_click. feedback_types={feedback_types}")
    
        click_i = idx["is_click"]
    
        response = torch.zeros_like(p)
    
        click_p = p[..., click_i]                      # (B, slate)
        click = torch.bernoulli(click_p).detach()      # (B, slate) 0/1
        response[..., click_i] = click
    
        gate = click  # 0/1 float tensor (B, slate)
    
        if "long_view" in idx:
            lv_i = idx["long_view"]
            lv_p = (p[..., lv_i] * gate).clamp(0.0, 1.0)
            response[..., lv_i] = torch.bernoulli(lv_p).detach()
    
        for name in ["is_like", "is_comment", "is_forward", "is_follow", "is_hate"]:
            if name in idx:
                i = idx[name]
                pp = (p[..., i] * gate).clamp(0.0, 1.0)
                response[..., i] = torch.bernoulli(pp).detach()

        flat = action.view(-1)
        step_cov_cnt = torch.unique(flat).numel()
        step_cov_ratio = step_cov_cnt / float(flat.numel())
        
        if not hasattr(self, "_cov_mask") or self._cov_mask.numel() != self.n_candidate:
            self._cov_mask = torch.zeros(self.n_candidate, dtype=torch.bool, device=action.device)
        self._cov_mask[flat] = True
        catalog_cov_ratio = self._cov_mask.float().mean().item()  # in [0,1]
                
        return {
            "immediate_response": response,
            "user_state": out_dict["state"],
            "coverage": step_cov_ratio,
            "ILD": 1 - torch.mean(corr_factor).item(),
           "coverage_ratio": float(step_cov_ratio), # : dedupe
           "catalog_coverage": float(catalog_cov_ratio), # : 
        }
        










    def get_ground_truth_user_state(self, profile, history):
        batch_data = {}
        batch_data.update(profile)
        batch_data.update(history)
        gt_state_dict = self.immediate_response_model.encode_state(batch_data, self.episode_batch_size)
        gt_user_state = gt_state_dict['state'].view(self.episode_batch_size, 1, self.gt_state_dim)
        return gt_user_state

    def get_intra_slate_similarity(self, action_item_encoding):
        '''
        @input:
        - action_item_encoding: (B, slate_size, enc_dim)
        @output:
        - similarity: (B, slate_size)
        '''
        B, L, d = action_item_encoding.shape
        pair_similarity = torch.mean(action_item_encoding.view(B, L, 1, d) * action_item_encoding.view(B, 1, L, d),
                                     dim=-1)
        point_similarity = torch.mean(pair_similarity, dim=-1)
        return point_similarity

    def _log_leave_debug(self, leave_prob, done_mask, response_dict):
        if not self.leave_debug_log_path or self.leave_debug_max_records <= 0:
            return
        if self._leave_debug_records_written >= self.leave_debug_max_records:
            return

        B = leave_prob.shape[0]
        user_ids = self.current_observation["user_profile"]["user_id"].detach().cpu().numpy().tolist()
        hist_len = self.current_observation["user_history"]["history_length"].detach().cpu().numpy().tolist()
        probs = leave_prob.detach().cpu().numpy().tolist()
        done = done_mask.detach().cpu().numpy().astype(np.int32).tolist()
        im = response_dict["immediate_response"].detach().cpu().numpy()
        # Aggregate current feedback over slate, keep per-feedback values.
        cur_fb = im.mean(axis=1).tolist()

        with open(self.leave_debug_log_path, "a", encoding="utf-8") as f:
            for i in range(B):
                if self._leave_debug_records_written >= self.leave_debug_max_records:
                    break
                rec = {
                    "step": int(self._leave_debug_step),
                    "batch_index": int(i),
                    "user_id": int(user_ids[i]),
                    "hist_len": int(hist_len[i]),
                    "leave_prob": float(probs[i]),
                    "leave_sampled": int(done[i]),
                    "current_feedback_mean": cur_fb[i],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self._leave_debug_records_written += 1

    def _build_leave_model_features(self, response_dict):
        """
        Build per-user features for learned leave-probability model.
        Feature layout must match train_exit_model.py:
          [user one-hot concat, hist_len_norm, hist_avg_resp..., prev1_resp...,
           recent-window resp means..., current_response..., engagement scalars]
        """
        B = self.current_observation["user_profile"]["user_id"].shape[0]
        device = self.device

        user_profile = self.current_observation["user_profile"]
        user_feats = []
        for f in self.observation_space.get("user_features", []):
            k = f"uf_{f}"
            if k in user_profile:
                user_feats.append(user_profile[k].float().view(B, -1))
        user_vec = torch.cat(user_feats, dim=1) if user_feats else torch.zeros(B, 0, device=device)

        history = self.current_observation["user_history"]
        hist_len = history["history_length"].float().view(B, 1)
        denom = torch.clamp(hist_len, min=1.0)
        hist_len_norm = (hist_len / float(max(1, self.max_hist_len))).clamp(0.0, 1.0)

        hist_rates = []
        prev1_rates = []
        recent_windows = self.leave_feature_spec.get("recent_windows", [5, 10])
        recent_rates = {int(w): [] for w in recent_windows}
        for r in self.response_types:
            k = f"history_{r}"
            if k in history:
                hr = history[k].float().view(B, -1)
                hist_rates.append((hr.sum(dim=1, keepdim=True) / denom).clamp(0.0, 1.0))
                prev1_rates.append(hr[:, -1:].clamp(0.0, 1.0))
                for w in recent_windows:
                    ww = max(1, min(int(w), hr.shape[1]))
                    d = torch.minimum(hist_len, torch.tensor(float(ww), device=device)).clamp(min=1.0)
                    recent = hr[:, -ww:].sum(dim=1, keepdim=True) / d
                    recent_rates[int(w)].append(recent.clamp(0.0, 1.0))
            else:
                hist_rates.append(torch.zeros(B, 1, device=device))
                prev1_rates.append(torch.zeros(B, 1, device=device))
                for w in recent_windows:
                    recent_rates[int(w)].append(torch.zeros(B, 1, device=device))
        hist_rate_vec = torch.cat(hist_rates, dim=1) if hist_rates else torch.zeros(B, 0, device=device)
        prev1_vec = torch.cat(prev1_rates, dim=1) if prev1_rates else torch.zeros(B, 0, device=device)
        recent_vecs = []
        for w in recent_windows:
            recent_vecs.append(torch.cat(recent_rates[int(w)], dim=1))
        recent_vec = torch.cat(recent_vecs, dim=1) if recent_vecs else torch.zeros(B, 0, device=device)

        cur_resp = response_dict["immediate_response"].float().mean(dim=1).view(B, -1)
        engage_weights = self.leave_feature_spec.get(
            "engage_weights", [1.0, 0.7, 0.5, 0.5, 0.5, 0.5, -0.2]
        )
        ew = torch.tensor(engage_weights, dtype=torch.float32, device=device).view(1, -1)
        ew = ew[:, : cur_resp.shape[1]]
        hist_score = (hist_rate_vec[:, : ew.shape[1]] * ew).sum(dim=1, keepdim=True)
        cur_score = (cur_resp[:, : ew.shape[1]] * ew).sum(dim=1, keepdim=True)
        score_delta = cur_score - hist_score

        feat = torch.cat(
            [user_vec, hist_len_norm, hist_rate_vec, prev1_vec, recent_vec, cur_resp, hist_score, cur_score, score_delta],
            dim=1,
        )
        expected_dim = int(self.leave_feature_spec["input_dim"])
        got_dim = int(feat.shape[1])
        if got_dim < expected_dim:
            pad = torch.zeros(B, expected_dim - got_dim, device=device)
            feat = torch.cat([feat, pad], dim=1)
        elif got_dim > expected_dim:
            feat = feat[:, :expected_dim]
        return feat

    def get_leave_signal(self, user_state, action, response_dict):
        '''
        User leave model maintains the user temper, and a user leaves when the temper drops below 1.
        @input:
        - user_state: not used in this env
        - action: not used in this env
        - response_dict: (B, slate_size, n_feedback)
        @process:
        - update temper
        @output:
        - done_mask:
        '''
        if self.leave_model is not None:
            with torch.no_grad():
                x = self._build_leave_model_features(response_dict)
                logits = self.leave_model(x)
                temp = max(1e-6, float(self.leave_logit_temperature))
                logits = logits / temp + float(self.leave_logit_bias)
                leave_prob = torch.sigmoid(logits)

                # Early-step probability ramp: step=0 uses leave_early_prob_scale, then linearly increases to 1.0.
                if self.leave_early_step_max >= 0 and self.leave_early_prob_scale < 1.0:
                    step = self.current_step.float()
                    denom = float(max(1, self.leave_early_step_max))
                    frac = (step / denom).clamp(0.0, 1.0)
                    scale = float(self.leave_early_prob_scale) + (1.0 - float(self.leave_early_prob_scale)) * frac
                    leave_prob = leave_prob * scale

                if self.leave_min_survival_steps > 0:
                    step = self.current_step.float()
                    guard = step < float(self.leave_min_survival_steps)
                    leave_prob = torch.where(guard, torch.zeros_like(leave_prob), leave_prob)

                leave_prob = leave_prob.clamp(1e-6, self.leave_prob_clip)
                done_mask = torch.bernoulli(leave_prob).bool()
                self.current_temper = (1.0 - leave_prob) * float(self.initial_temper)
                self._log_leave_debug(leave_prob, done_mask, response_dict)
                self._leave_debug_step += 1
            return done_mask

        temper_boost = self._aggregate_temper_reward(response_dict)
        temper_update = temper_boost - self.temper_cost_offset
        temper_update[temper_update > 0] = 0
        temper_update[temper_update < -float(self.temper_max_drop)] = -float(self.temper_max_drop)
        self.current_temper += temper_update
        done_mask = self.current_temper < 1
        return done_mask

    def update_observation(self, user_state, action, response, done_mask, update_current=True):
        '''
        user profile stays static, only update user history
        @input:
        - user_state: not used in this env
        - action: (B, slate_size), indices of self.candidate_iids
        - response: (B, slate_size, n_feedback)
        - done_mask: not used in this env
        @output:
        - update_info: {slate: (B, slate_size),
                        updated_observation: same format as self.reset@output - observation}
        '''
        if action.dim() == 1:
            action = action.view(-1, 1)
        if response.dim() == 2:
            response = response.unsqueeze(1)

        consumed_len = self._get_history_consume_length(action.shape[1])
        self._last_history_consume_count = consumed_len
        action = action[:, :consumed_len]
        response = response[:, :consumed_len, :]

        rec_list = self.candidate_iids[action]

        old_history = self.current_observation['user_history']
        max_H = self.max_hist_len
        L = old_history['history_length'] + consumed_len
        L[L > max_H] = max_H
        new_history = {'history': torch.cat((old_history['history'], rec_list), dim=1)[:, -max_H:],
                       'history_length': L}
        for k, candidate_meta_features in self.candidate_item_meta.items():
            meta_features = candidate_meta_features[action]
            previous_meta = old_history[f'history_{k}'].view(self.episode_batch_size, max_H, -1)
            new_history[f'history_{k}'] = torch.cat((previous_meta, meta_features), dim=1)[:, -max_H:, :].view(
                self.episode_batch_size, -1)
        for i, R in enumerate(self.immediate_response_model.feedback_types):
            k = f'history_{R}'
            new_history[k] = torch.cat((old_history[k], response[:, :, i]), dim=1)[:, -max_H:]
        if update_current:
            self.current_observation['user_history'] = new_history
        return {'slate': rec_list, 'consumed_slate_size': consumed_len, 'updated_observation': {
            'user_profile': deepcopy(self.current_observation['user_profile']),
            'user_history': deepcopy(new_history)}}

    def sample_new_batch_from_reader(self):
        '''
        @output
        - sample_info: see BaseRLEnvironment.get_observation_from_batch@input - sample_batch
        '''
        new_sample_flag = False
        try:
            sample_info = next(self.batch_iter)
            if sample_info['user_profile'].shape[0] != self.episode_batch_size:
                new_sample_flag = True
        except:
            new_sample_flag = True
        if new_sample_flag:
            self.batch_iter = iter(DataLoader(self.reader, batch_size=self.episode_batch_size, shuffle=True,
                                              pin_memory=True, num_workers=8))
            sample_info = next(self.batch_iter)
        return sample_info

    def stop(self):
        self.batch_iter = None

    def get_new_iterator(self, B):
        return iter(DataLoader(self.reader, batch_size=B, shuffle=True,
                               pin_memory=True, num_workers=8))

    def create_observation_buffer(self, buffer_size):
        '''
        @input:
        - buffer_size: L, scalar
        @output:
        - observation: {'user_profile': {'user_id': (L,),
                                         'uf_{feature_name}': (L, feature_dim)},
                        'user_history': {'history': (L, max_H),
                                         'history_if_{feature_name}': (L, max_H * feature_dim),
                                         'history_{response}': (L, max_H),
                                         'history_length': (L,)}}
        '''
        observation = {'user_profile': {'user_id': torch.zeros(buffer_size).to(torch.long).to(self.device)},
                       'user_history': {
                           'history': torch.zeros(buffer_size, self.max_hist_len).to(torch.long).to(self.device),
                           'history_length': torch.zeros(buffer_size).to(torch.long).to(self.device)}}
        for f, f_dim in self.observation_space['user_feature_dims'].items():
            observation['user_profile'][f'uf_{f}'] = torch.zeros(buffer_size, f_dim).to(torch.float).to(self.device)
        for f, f_dim in self.observation_space['item_feature_dims'].items():
            observation['user_history'][f'history_if_{f}'] = torch.zeros(buffer_size, f_dim * self.max_hist_len)\
                .to(torch.float).to(self.device)
        for f in self.observation_space['feedback_type']:
            observation['user_history'][f'history_{f}'] = torch.zeros(buffer_size, self.max_hist_len)\
                .to(torch.float).to(self.device)
        return observation

    def get_report(self, smoothness=10):
        return {k: np.mean(v[-smoothness:]) for k, v in self.env_history.items()}
