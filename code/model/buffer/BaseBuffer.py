import torch
import torch.nn.functional as F
import random
import numpy as np

import utils

class BaseBuffer():
    '''
    The general buffer
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - buffer_size
        '''
        parser.add_argument('--buffer_size', type=int, default=10000,
                            help='replay buffer size')
        return parser
        
    def __init__(self, *input_args):
        args, env, policy, critic = input_args
        self.buffer_size = args.buffer_size
        super().__init__()
        self.device = args.device
        self.buffer_head = 0
        self.current_buffer_size = 0
        self.n_stream_record = 0
        
    def reset(self, *reset_args):
        '''
        @output:
        - buffer: {'observation': {'user_profile': {'user_id': (L,), 
                                                    'uf_{feature_name}': (L, feature_dim)}, 
                                   'user_history': {'history': (L, max_H), 
                                                    'history_if_{feature_name}': (L, max_H * feature_dim), 
                                                    'history_{response}': (L, max_H), 
                                                    'history_length': (L,)}}
                   'policy_output': {'state': (L, state_dim), 
                                     'action': (L, action_dim), 
                                     'prob': (L, slate_size)}, 
                   'next_observation': same format as @output-buffer['observation'], 
                   'done_mask': (L,),
                   'response': {'reward': (L,), 
                                'immediate_response':, (L, slate_size * response_dim)}}
        '''
        env = reset_args[0]
        actor = reset_args[1]
        observation = env.create_observation_buffer(self.buffer_size)
        next_observation = env.create_observation_buffer(self.buffer_size)
        policy_output = {'state': torch.zeros(self.buffer_size, actor.state_dim)\
                                         .to(torch.float).to(self.device), 
                         'action': torch.zeros(self.buffer_size, actor.action_dim)\
                                         .to(torch.long).to(self.device), 
                         'prob': torch.zeros(self.buffer_size, env.slate_size)\
                                         .to(torch.float).to(self.device)}
        reward = torch.zeros(self.buffer_size).to(torch.float).to(self.device)
        done = torch.zeros(self.buffer_size).to(torch.bool).to(self.device)
        im_response = torch.zeros(self.buffer_size, env.response_dim * env.slate_size)\
                                         .to(torch.float).to(self.device)
        self.buffer = {'observation': observation, 
                       'policy_output': policy_output, 
                       'user_response': {'reward': reward, 'immediate_response': im_response},
                       'done_mask': done,
                       'next_observation': next_observation}
        return self.buffer
    
    
    def sample(self, batch_size):
        '''
        Batch sample is organized as a tuple of (observation, policy_output, user_response, done_mask, next_observation)
        
        Buffer: see reset@output
        @output:
        - observation: {'user_profile': {'user_id': (B,), 
                                         'uf_{feature_name}': (B, feature_dim)}, 
                        'user_history': {'history': (B, max_H), 
                                         'history_if_{feature_name}': (B, max_H * feature_dim), 
                                         'history_{response}': (B, max_H), 
                                         'history_length': (B,)}}
        - policy_output: {'state': (B, state_dim), 
                          'action': (B, slate_size), 
                          'prob': (B, slate_size)}, 
        - user_feedback: {'reward': (B,), 
                          'immediate_response':, (B, slate_size * response_dim)}}
        - done_mask: (B,),
        - next_observation: same format as @output - observation, 
        '''
        indices = np.random.randint(0, self.current_buffer_size, size = batch_size)
        profile = {k:v[indices] for k,v in self.buffer["observation"]["user_profile"].items()}
        history = {k:v[indices] for k,v in self.buffer["observation"]["user_history"].items()}
        observation = {"user_profile": profile, "user_history": history}
        profile = {k:v[indices] for k,v in self.buffer["next_observation"]["user_profile"].items()}
        history = {k:v[indices] for k,v in self.buffer["next_observation"]["user_history"].items()}
        next_observation = {"user_profile": profile, "user_history": history}
        policy_output = {"state": self.buffer["policy_output"]["state"][indices], 
                         "action": self.buffer["policy_output"]["action"][indices], 
                         "prob": self.buffer["policy_output"]["prob"][indices]}
        user_response = {"reward": self.buffer["user_response"]["reward"][indices], 
                         "immediate_response": self.buffer["user_response"]["immediate_response"][indices]}
        done_mask = self.buffer["done_mask"][indices]
        return observation, policy_output, user_response, done_mask, next_observation
    
    def update(self, observation, policy_output, user_feedback, next_observation):
        '''
        @input:
        - observation: {'user_profile': {'user_id': (B,), 
                                         'uf_{feature_name}': (B, feature_dim)}, 
                        'user_history': {'history': (B, max_H), 
                                         'history_if_{feature_name}': (B, max_H * feature_dim), 
                                         'history_{response}': (B, max_H), 
                                         'history_length': (B,)}}
        - policy_output: {'user_state': (B, state_dim), 
                          'prob': (B, action_dim),
                          'action': (B, action_dim)}
        - user_feedback: {'done': (B,), 
                          'immdiate_response':, (B, action_dim * feedback_dim), 
                          'reward': (B,)}
        - next_observation: same format as update_buffer@input-observation
        '''
        B = len(user_feedback['reward'])
        if self.buffer_head + B >= self.buffer_size:
            tail = self.buffer_size - self.buffer_head
            indices = [self.buffer_head + i for i in range(tail)] + \
                        [i for i in range(B - tail)]
        else:
            indices = [self.buffer_head + i for i in range(B)]
        indices = torch.tensor(indices).to(torch.long).to(self.device)

        def _cast_like(dst, src):
            if torch.is_tensor(src):
                return src.to(device=dst.device, dtype=dst.dtype)
            return torch.as_tensor(src, device=dst.device, dtype=dst.dtype)
        
        for k,v in observation['user_profile'].items():
            dst = self.buffer['observation']['user_profile'][k]
            dst[indices] = _cast_like(dst, v)
        for k,v in observation['user_history'].items():
            dst = self.buffer['observation']['user_history'][k]
            dst[indices] = _cast_like(dst, v)
        for k,v in next_observation['user_profile'].items():
            dst = self.buffer['next_observation']['user_profile'][k]
            dst[indices] = _cast_like(dst, v)
        for k,v in next_observation['user_history'].items():
            dst = self.buffer['next_observation']['user_history'][k]
            dst[indices] = _cast_like(dst, v)
        self.buffer['policy_output']['state'][indices] = _cast_like(self.buffer['policy_output']['state'], policy_output['state'])
        self.buffer['policy_output']['action'][indices] = _cast_like(self.buffer['policy_output']['action'], policy_output['action'])
        self.buffer['policy_output']['prob'][indices] = _cast_like(self.buffer['policy_output']['prob'], policy_output['prob'])
        self.buffer['user_response']['immediate_response'][indices] = _cast_like(
            self.buffer['user_response']['immediate_response'],
            user_feedback['immediate_response'].view(B, -1),
        )
        self.buffer['user_response']['reward'][indices] = _cast_like(self.buffer['user_response']['reward'], user_feedback['reward'])
        self.buffer['done_mask'][indices] = _cast_like(self.buffer['done_mask'], user_feedback['done'])
        
        self.buffer_head = (self.buffer_head + B) % self.buffer_size
        self.n_stream_record += B
        self.current_buffer_size = min(self.n_stream_record, self.buffer_size)
        
